from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RGNBConfig:
    ne: int = 8
    np: int = 1024
    m_steps: int = 100
    eta: float = 5e-3
    l0: float = 1.0
    delta: float = 0.5
    gp_noise: float = 1e-3
    bridge_T: int = 100
    bridge_epochs: int = 100
    bridge_lr: float = 1e-3
    top_k: int = 128
    cfg_scale: float = 2.0
    lambda_rank: float = 0.2
    lambda_manifold: float = 0.1
    vae_latent_dim: int = 16
    device: str = "cpu"


@dataclass
class SyntheticPair:
    x_t: torch.Tensor  # source (low value)
    y_t: torch.Tensor
    x_0: torch.Tensor  # target (high value)
    y_0: torch.Tensor


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPPosteriorMeanSampler:
    """Algorithm 1: fit GP, sample kernel scales, and do bidirectional optimization."""

    def __init__(self, config: RGNBConfig):
        self.config = config
    def _fit_gp(self, x: torch.Tensor, y: torch.Tensor, lengthscale: float) -> Tuple[ExactGPModel, gpytorch.likelihoods.GaussianLikelihood]:
        # 确保 GP 模型与训练数据在同一设备上（避免 cuda/cpu 混用）
        device = x.device
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGPModel(x, y, likelihood).to(device)
        model.covar_module.base_kernel.lengthscale = lengthscale
        likelihood.noise = torch.as_tensor(self.config.gp_noise, device=device)

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(50):
            optimizer.zero_grad()
            output = model(x)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()
        model.eval()
        likelihood.eval()
        return model, likelihood

    def _posterior_mean(self, model: ExactGPModel, likelihood: gpytorch.likelihoods.GaussianLikelihood, x: torch.Tensor) -> torch.Tensor:
        with gpytorch.settings.fast_pred_var():
            posterior = likelihood(model(x))
        return posterior.mean

    def _gradient_optimize(self, model: ExactGPModel, likelihood: gpytorch.likelihoods.GaussianLikelihood, x0: torch.Tensor, ascent: bool) -> torch.Tensor:
        x = x0.clone().detach().requires_grad_(True)
        for _ in range(self.config.m_steps):
            mean = self._posterior_mean(model, likelihood, x)
            grad = torch.autograd.grad(mean.sum(), x, create_graph=False, retain_graph=False)[0]
            direction = 1.0 if ascent else -1.0
            x = (x + direction * self.config.eta * grad).detach().requires_grad_(True)
        return x.detach()

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> List[SyntheticPair]:
        n = x.size(0)
        all_pairs: List[SyntheticPair] = []
        for _ in range(self.config.ne):
            l = torch.empty(1).uniform_(self.config.l0 - self.config.delta, self.config.l0 + self.config.delta).item()
            model, likelihood = self._fit_gp(x, y, l)
            idx = torch.randint(0, n, (min(self.config.np, n),), device=x.device)
            starts = x[idx]
            x_plus = self._gradient_optimize(model, likelihood, starts, ascent=True)
            x_minus = self._gradient_optimize(model, likelihood, starts, ascent=False)
            y_plus = self._posterior_mean(model, likelihood, x_plus).detach()
            y_minus = self._posterior_mean(model, likelihood, x_minus).detach()
            for j in range(x_plus.size(0)):
                all_pairs.append(SyntheticPair(x_minus[j], y_minus[j], x_plus[j], y_plus[j]))
        return all_pairs


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class RankNet(nn.Module):
    """3-layer MLP with pairwise ranking loss support."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @staticmethod
    def pairwise_hinge_loss(s_pos: torch.Tensor, s_neg: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
        return F.relu(margin - (s_pos - s_neg)).mean()

    @staticmethod
    def pairwise_bce_loss(s_pos: torch.Tensor, s_neg: torch.Tensor) -> torch.Tensor:
        logits = s_pos - s_neg
        labels = torch.ones_like(logits)
        return F.binary_cross_entropy_with_logits(logits, labels)


class ScoreNetwork(nn.Module):
    """4-layer MLP with Swish, input=(x_t,t,y_cond)."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        in_dim = dim + 2  # t scalar + y_cond scalar
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            Swish(),
            nn.Linear(hidden, hidden),
            Swish(),
            nn.Linear(hidden, hidden),
            Swish(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y_cond: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if y_cond.ndim == 1:
            y_cond = y_cond.unsqueeze(-1)
        h = torch.cat([x_t, t, y_cond], dim=-1)
        return self.net(h)


class ManifoldVAE(nn.Module):
    def __init__(self, dim: int, latent_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss_fn(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1e-3) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl

    def log_density_proxy(self, x: torch.Tensor) -> torch.Tensor:
        recon, _, _ = self.forward(x)
        return -((x - recon) ** 2).sum(dim=-1)


class BrownianBridgeTrainer:
    def __init__(self, score_net: ScoreNetwork, config: RGNBConfig):
        self.score_net = score_net
        self.config = config

    def kappa(self, t: torch.Tensor) -> torch.Tensor:
        # simple variance schedule
        s = t / float(self.config.bridge_T)
        return 0.1 * s * (1 - s) + 1e-4

    def train(self, pairs: List[SyntheticPair]):
        if not pairs:
            return
        x_tgt = torch.stack([p.x_0 for p in pairs])
        x_src = torch.stack([p.x_t for p in pairs])
        y_cond = torch.stack([p.y_0 for p in pairs]).float()

        optimizer = torch.optim.Adam(self.score_net.parameters(), lr=self.config.bridge_lr)
        self.score_net.train()
        n = x_tgt.size(0)

        for _ in range(self.config.bridge_epochs):
            idx = torch.randint(0, n, (min(512, n),), device=x_tgt.device)
            x0 = x_tgt[idx]
            xT = x_src[idx]
            y = y_cond[idx]
            t = torch.randint(1, self.config.bridge_T + 1, (x0.size(0),), device=x0.device).float()
            mt = 1.0 - t / float(self.config.bridge_T)
            kappa_tt = self.kappa(t)
            eps = torch.randn_like(x0)
            xt = mt.unsqueeze(-1) * x0 + (1 - mt).unsqueeze(-1) * xT + torch.sqrt(kappa_tt).unsqueeze(-1) * eps
            target = mt.unsqueeze(-1) * (xT - x0) + torch.sqrt(kappa_tt).unsqueeze(-1) * eps

            eps_hat = self.score_net(xt, t / float(self.config.bridge_T), y)
            loss = F.mse_loss(eps_hat, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class RGNBModel:
    def __init__(self, dim: int, config: RGNBConfig | None = None):
        self.config = config or RGNBConfig()
        self.rank_net = RankNet(dim).to(self.config.device)
        self.score_net = ScoreNetwork(dim).to(self.config.device)
        self.vae = ManifoldVAE(dim, latent_dim=self.config.vae_latent_dim).to(self.config.device)
        self.synth = GPPosteriorMeanSampler(self.config)
        self.bridge = BrownianBridgeTrainer(self.score_net, self.config)

    def train_ranknet(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 50, use_bce: bool = True):
        x, y = x.to(self.config.device), y.to(self.config.device)
        opt = torch.optim.Adam(self.rank_net.parameters(), lr=1e-3)
        n = x.size(0)
        self.rank_net.train()
        for _ in range(epochs):
            i = torch.randint(0, n, (min(512, n),), device=x.device)
            j = torch.randint(0, n, (min(512, n),), device=x.device)
            xi, xj = x[i], x[j]
            yi, yj = y[i], y[j]
            s_i, s_j = self.rank_net(xi), self.rank_net(xj)

            pos_mask = yi >= yj
            s_pos = torch.where(pos_mask, s_i, s_j)
            s_neg = torch.where(pos_mask, s_j, s_i)
            loss = RankNet.pairwise_bce_loss(s_pos, s_neg) if use_bce else RankNet.pairwise_hinge_loss(s_pos, s_neg)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def train_vae(self, x: torch.Tensor, epochs: int = 50):
        x = x.to(self.config.device)
        opt = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        self.vae.train()
        n = x.size(0)
        for _ in range(epochs):
            idx = torch.randint(0, n, (min(512, n),), device=x.device)
            xb = x[idx]
            recon, mu, logvar = self.vae(xb)
            loss = ManifoldVAE.loss_fn(recon, xb, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def train_bridge(self, x: torch.Tensor, y: torch.Tensor):
        pairs = self.synth.generate(x.to(self.config.device), y.to(self.config.device))
        self.bridge.train(pairs)

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        self.train_ranknet(x, y)
        self.train_vae(x)
        self.train_bridge(x, y)

    def _cfg_score(self, xt: torch.Tensor, t_scaled: torch.Tensor, y_cond: torch.Tensor) -> torch.Tensor:
        eps_cond = self.score_net(xt, t_scaled, y_cond)
        eps_uncond = self.score_net(xt, t_scaled, torch.zeros_like(y_cond))
        return eps_uncond + self.config.cfg_scale * (eps_cond - eps_uncond)

    def sample(self, x_offline: torch.Tensor, y_offline: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        self.rank_net.eval()
        self.vae.eval()
        self.score_net.eval()

        steps = steps or self.config.bridge_T
        device = self.config.device
        x_offline = x_offline.to(device)
        y_offline = y_offline.to(device)

        top_idx = torch.argsort(y_offline, descending=True)[: min(self.config.top_k, x_offline.size(0))]
        x = x_offline[top_idx].clone().detach().requires_grad_(True)
        target_y = y_offline[top_idx]

        for t in range(steps, 0, -1):
            t_scaled = torch.full((x.size(0), 1), t / float(steps), device=device)
            kappa_t = self.bridge.kappa(torch.full((x.size(0),), float(t), device=device))
            score = self._cfg_score(x, t_scaled, target_y)
            mu_theta = x - (1.0 / steps) * score

            rank_score = self.rank_net(x).sum()
            g_rank = torch.autograd.grad(rank_score, x, retain_graph=True, create_graph=True)[0]

            dens = self.vae.log_density_proxy(x).sum()
            g_mani = torch.autograd.grad(dens, x, retain_graph=True, create_graph=True)[0]

            noise = torch.randn_like(x) * torch.sqrt(kappa_t).unsqueeze(-1)
            x = (mu_theta + self.config.lambda_rank * g_rank + self.config.lambda_manifold * g_mani + noise).requires_grad_(True)

        return x.detach()
