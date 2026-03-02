from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from .root_bb import create_root_brownian_bridge


@dataclass
class RGNBConfig:
    ne: int = 8
    np: int = 1024
    m_steps: int = 100
    eta: float = 5e-3
    l0: float = 1.0
    delta: float = 0.5
    gp_noise: float = 1e-3
    bridge_T: int = 200
    bridge_epochs: int = 200
    bridge_lr: float = 1e-3
    top_k: int = 128
    cfg_scale: float = 2.0
    lambda_rank: float = 0.05
    lambda_manifold: float = 0.1
    cfg_dropout: float = 0.15
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

    def _gradient_optimize(
        self,
        model: ExactGPModel,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        x0: torch.Tensor,
        ascent: bool,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
    ) -> torch.Tensor:
        x = x0.clone().detach().requires_grad_(True)
        for _ in range(self.config.m_steps):
            mean = self._posterior_mean(model, likelihood, x)
            grad = torch.autograd.grad(mean.sum(), x, create_graph=False, retain_graph=False)[0]
            direction = 1.0 if ascent else -1.0
            x_new = x + direction * self.config.eta * grad
            # clip to the empirical design domain to avoid exploding designs
            x_new = torch.max(torch.min(x_new, x_max), x_min)
            x = x_new.detach().requires_grad_(True)
        return x.detach()

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> List[SyntheticPair]:
        n = x.size(0)
        all_pairs: List[SyntheticPair] = []
        # empirical bounds of the (normalized) design space for clipping
        x_min = x.min(dim=0).values
        x_max = x.max(dim=0).values
        for _ in range(self.config.ne):
            l = torch.empty(1).uniform_(self.config.l0 - self.config.delta, self.config.l0 + self.config.delta).item()
            model, likelihood = self._fit_gp(x, y, l)
            idx = torch.randint(0, n, (min(self.config.np, n),), device=x.device)
            starts = x[idx]
            x_plus = self._gradient_optimize(model, likelihood, starts, ascent=True, x_min=x_min, x_max=x_max)
            x_minus = self._gradient_optimize(model, likelihood, starts, ascent=False, x_min=x_min, x_max=x_max)
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

    def train(self, pairs: List[SyntheticPair], dropout_prob: float | None = None):
        if not pairs:
            return
        x_tgt = torch.stack([p.x_0 for p in pairs])
        x_src = torch.stack([p.x_t for p in pairs])
        y_cond = torch.stack([p.y_0 for p in pairs]).float()

        optimizer = torch.optim.Adam(self.score_net.parameters(), lr=self.config.bridge_lr)
        self.score_net.train()
        n = x_tgt.size(0)

        if dropout_prob is None:
            dropout_prob = getattr(self.config, "cfg_dropout", 0.15)

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

            # classifier-free style conditioning dropout on y
            if dropout_prob > 0.0:
                mask = torch.rand_like(y) < dropout_prob
                y_train = y.clone()
                y_train[mask] = 0.0
            else:
                y_train = y

            eps_hat = self.score_net(xt, t / float(self.config.bridge_T), y_train)
            loss = F.mse_loss(eps_hat, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class RGNBModel:
    def __init__(self, dim: int, config: RGNBConfig | None = None):
        self.config = config or RGNBConfig()
        # self.rank_net = RankNet(dim).to(self.config.device)
        # self.score_net = ScoreNetwork(dim).to(self.config.device)
        # self.vae = ManifoldVAE(dim, latent_dim=self.config.vae_latent_dim).to(self.config.device)
        self.synth = GPPosteriorMeanSampler(self.config)

        # ROOT Brownian Bridge 模型（与 ROOT 完全一致的训练 / 采样逻辑）
        self.bb_model = create_root_brownian_bridge(dim).to(self.config.device)
        self.bb_optimizer = torch.optim.Adam(self.bb_model.get_parameters(), lr=self.config.bridge_lr)

    def train_ranknet(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 50, use_bce: bool = True):
        x, y = x.to(self.config.device), y.to(self.config.device)
        opt = torch.optim.Adam(self.rank_net.parameters(), lr=1e-3)
        n = x.size(0)
        self.rank_net.train()
        for epoch in range(epochs):
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

            # 简单调试：每若干个 epoch 打印一次 pairwise 排序准确率
            if (epoch + 1) % max(1, epochs // 5) == 0:
                with torch.no_grad():
                    i_dbg = torch.randint(0, n, (min(2048, n),), device=x.device)
                    j_dbg = torch.randint(0, n, (min(2048, n),), device=x.device)
                    xi_dbg, xj_dbg = x[i_dbg], x[j_dbg]
                    yi_dbg, yj_dbg = y[i_dbg], y[j_dbg]
                    s_i_dbg = self.rank_net(xi_dbg)
                    s_j_dbg = self.rank_net(xj_dbg)
                    acc = ((yi_dbg >= yj_dbg) == (s_i_dbg >= s_j_dbg)).float().mean().item()
                    print(f"[RankNet][epoch {epoch+1}/{epochs}] loss={loss.item():.4f}, pairwise_acc≈{acc:.3f}")

    def train_vae(self, x: torch.Tensor, epochs: int = 50):
        x = x.to(self.config.device)
        opt = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        self.vae.train()
        n = x.size(0)
        running = 0.0
        for epoch in range(epochs):
            idx = torch.randint(0, n, (min(512, n),), device=x.device)
            xb = x[idx]
            recon, mu, logvar = self.vae(xb)
            loss = ManifoldVAE.loss_fn(recon, xb, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

            if (epoch + 1) % max(1, epochs // 5) == 0:
                avg = running / max(1, epochs // 5)
                print(f"[VAE][epoch {epoch+1}/{epochs}] loss≈{avg:.4f}")
                running = 0.0

    def train_bridge(self, x: torch.Tensor, y: torch.Tensor):
        """
        使用 ROOT 的 BrownianBridgeModel 在 GP 合成的 (x_high, x_low, y_high, y_low) 上训练。
        """
        device = self.config.device
        x = x.to(device)
        y = y.to(device)

        self.bb_model.train()
        pairs = self.synth.generate(x, y)
        for epoch in range(self.config.bridge_epochs):
            # 每个 epoch 从 GP 重新合成一批 pairs，贴近 ROOT 的做法
            

            if not pairs:
                print("[Bridge][ROOT] WARNING: no synthetic pairs generated, bridge will be skipped.")
                return

            x_high = torch.stack([p.x_0 for p in pairs]).to(device)
            x_low = torch.stack([p.x_t for p in pairs]).to(device)
            y_high = torch.stack([p.y_0 for p in pairs]).float().to(device).unsqueeze(-1)
            y_low = torch.stack([p.y_t for p in pairs]).float().to(device).unsqueeze(-1)

            loss, _ = self.bb_model(x_high, y_high, x_low, y_low)
            self.bb_optimizer.zero_grad()
            loss.backward()
            self.bb_optimizer.step()

            if (epoch + 1) % max(1, self.config.bridge_epochs // 5) == 0:
                print(f"[Bridge][ROOT][epoch {epoch+1}/{self.config.bridge_epochs}] loss≈{loss.item():.4f}")

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        self.train_ranknet(x, y)
        self.train_vae(x)
        self.train_bridge(x, y)

    def sample(self, x_offline: torch.Tensor, y_offline: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        """
        使用 ROOT 的 BrownianBridgeModel 做采样：
        - 从离线数据中选取一批低分候选 (x_low, y_low)
        - 构造统一的高分条件 y_high
        - 调用 bb_model.sample 得到高分候选
        """
        device = self.config.device
        self.bb_model.eval()

        x_offline = x_offline.to(device)
        y_offline = y_offline.to(device)

        # 选取 top_k 个低分样本作为起点（与 ROOT 的 "low candidates" 一致）
        n = x_offline.size(0)
        k = min(self.config.top_k, n)
        low_idx = torch.argsort(y_offline, descending=False)[:k]
        x_low = x_offline[low_idx]
        y_low = y_offline[low_idx].unsqueeze(-1)

        # 目标高分：简单使用当前 offline 的最大值作为 oracle proxy
        target_y_val = y_offline.max()
        y_high = torch.full_like(y_low, target_y_val)

        print(
            f"[Sample][ROOT] low candidates: mean={y_low.mean().item():.4f}, "
            f"min={y_low.min().item():.4f}, max={y_low.max().item():.4f}; "
            f"target_y={target_y_val.item():.4f}"
        )

        with torch.no_grad():
            high_candidates = self.bb_model.sample(
                x_low,
                y_low,
                y_high,
                clip_denoised=False,
                sample_mid_step=False,
                classifier_free_guidance_weight=0.0,
            )

        return high_candidates
