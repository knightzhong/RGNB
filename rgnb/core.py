from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
    # 目标分外推倍率：仅在未提供 high_cond_score 时使用
    target_extrap_factor: float = 1.0
    # 引导仅在采样后期施加：guidance_start_ratio=0.5 表示仅在后 50% 的步骤中注入引导
    guidance_start_ratio: float = 0.5
    # 额外 GP 相关配置（仿 ROOT，实现上简化）
    gp_max_fit: int = 4096        # 拟合 GP 时最多使用的训练样本数，防止 OOM
    gp_threshold_diff: float = 1e-3
    # 起点选择策略：对齐 ROOT 的 type_of_initial_points（highest / lowest / all）
    gp_initial_points_type: str = "all"
    vae_latent_dim: int = 16
    device: str = "cpu"
    # 与 ROOT use_fixed_gp_posterior 对齐：固定种子生成一次对、val_frac、小批量
    gp_seed: int = 0
    val_frac: float = 0.1
    bridge_batch_size: int = 64
    # EMA（与 ROOT 一致，测试时用 EMA 权重采样，常能提升 1～3 个点）
    use_ema: bool = True
    ema_decay: float = 0.995
    start_ema_step: int = 4000
    update_ema_interval: int = 8


@dataclass
class SyntheticPair:
    x_t: torch.Tensor  # source (low value)
    y_t: torch.Tensor
    x_0: torch.Tensor  # target (high value)
    y_0: torch.Tensor


class PairsDataset(Dataset):
    """((x_high, y_high), (x_low, y_low)) 格式，与 ROOT create_train_dataloader 一致。"""

    def __init__(self, pairs: List[SyntheticPair]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        return (p.x_0, p.y_0.view(1)), (p.x_t, p.y_t.view(1))


class EMA:
    """指数移动平均，与 ROOT runners.base.EMA 一致，测试时用 EMA 权重采样。"""

    def __init__(self, ema_decay: float):
        self.ema_decay = ema_decay
        self.backup: dict = {}
        self.shadow: dict = {}

    def register(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module, with_decay: bool = True) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if with_decay:
                    new_avg = (1.0 - self.ema_decay) * param.data + self.ema_decay * self.shadow[name]
                else:
                    new_avg = param.data.clone()
                self.shadow[name] = new_avg

    def apply_shadow(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


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

    def generate(self, x: torch.Tensor, y: torch.Tensor) -> List[SyntheticPair]:
        """
        单 GP 版本（对齐 ROOT 的 sampling_data_from_fixed_GP 思路）：
        - 只拟合一次 GP 后验均值函数（固定 lengthscale / variance，不在每个函数内扰动）；
        - 对每个“函数”在同一个 GP 上做梯度下降/上升；
        - 过滤 high_y - low_y <= threshold_diff 的 pair。
        - 使用 gp_seed 固定随机种子，与 ROOT sampling_data_from_fixed_GP(..., seed=0) 一致。
        """
        from gaussian_process.GP import GP as RootGP  # ROOT 中的 GP 实现

        gp_seed = getattr(self.config, "gp_seed", 0)
        torch.manual_seed(gp_seed)

        device = x.device
        n, d = x.size()

        # 子采样用于拟合 GP，避免 OOM（仅影响 GP 拟合，用于梯度优化的起点池单独控制）
        max_fit = min(self.config.gp_max_fit, n)
        if n > max_fit:
            idx_fit = torch.randperm(n, device=device)[:max_fit]
            x_train = x[idx_fit]
            y_train = y[idx_fit]
        else:
            x_train, y_train = x, y

        # 起点池：对齐 ROOT 的 type_of_initial_points 逻辑
        # - "highest": 仅从高分样本中构造 best_x；
        # - "lowest": 仅从低分样本中构造；
        # - 其它：退化为使用全部样本。
        y_flat = y.view(-1)
        init_type = getattr(self.config, "gp_initial_points_type", "all")
        if init_type == "highest":
            k = min(self.config.np, n)
            idx_pool = torch.argsort(y_flat)[-k:]
            x_pool = x[idx_pool]
        elif init_type == "lowest":
            k = min(self.config.np, n)
            idx_pool = torch.argsort(y_flat)[:k]
            x_pool = x[idx_pool]
        else:
            x_pool = x

        # 初始超参数（仿 ROOT：initial_lengthscale / initial_outputscale / noise / mean_prior）
        lengthscale = torch.tensor(self.config.l0, device=device)
        variance = torch.tensor(1.0, device=device)
        noise = torch.tensor(self.config.gp_noise, device=device)
        mean_prior = torch.tensor(0.0, device=device)

        gp_model = RootGP(
            device=device,
            x_train=x_train,
            y_train=y_train,
            lengthscale=lengthscale,
            variance=variance,
            noise=noise,
            mean_prior=mean_prior,
        )

        num_functions = self.config.ne
        num_points = min(self.config.np, x_pool.size(0))
        num_gradient_steps = self.config.m_steps
        lr = self.config.eta
        threshold_diff = self.config.gp_threshold_diff

        # 学习率向量：[低分点用 -lr，高分点用 +lr]
        learning_rate_vec = torch.cat(
            (
                -lr * torch.ones(num_points, d, device=device),
                lr * torch.ones(num_points, d, device=device),
            )
        )

        # 固定 GP 超参数：只在初始 lengthscale / variance 上拟合一次（单 GP）
        gp_model.set_hyper(lengthscale=lengthscale, variance=variance)

        all_pairs: List[SyntheticPair] = []

        for _ in range(num_functions):

            # 从起点池 x_pool 中随机选 num_points 个起点（对齐 ROOT: best_x -> sampling_data_from_fixed_GP）
            sel_idx = torch.randperm(x_pool.size(0), device=device)[:num_points]
            low_x = x_pool[sel_idx].clone().detach().requires_grad_(True)
            high_x = x_pool[sel_idx].clone().detach().requires_grad_(True)
            joint_x = torch.cat((low_x, high_x), dim=0)

            # 在固定 GP 上做梯度下降 / 上升以找到低值 / 高值设计
            for _ in range(num_gradient_steps):
                mu_star = gp_model.mean_posterior(joint_x)
                grad = torch.autograd.grad(mu_star.sum(), joint_x, create_graph=False, retain_graph=False)[0]
                joint_x = (joint_x + learning_rate_vec * grad).detach().requires_grad_(True)

            joint_y = gp_model.mean_posterior(joint_x).detach()

            low_x = joint_x[:num_points, :]
            high_x = joint_x[num_points:, :]
            low_y = joint_y[:num_points]
            high_y = joint_y[num_points:]

            for i in range(num_points):
                if high_y[i] - low_y[i] <= threshold_diff:
                    continue
                all_pairs.append(
                    SyntheticPair(
                        x_t=low_x[i].detach(),
                        y_t=low_y[i].detach(),
                        x_0=high_x[i].detach(),
                        y_0=high_y[i].detach(),
                    )
                )

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
        """确定性路径：仅用 mu 解码，不经过 reparameterize，避免采样时梯度抖动。"""
        mu, _ = self.encode(x)
        recon = self.decode(mu)
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
        # Rank & Manifold 模块：用于采样阶段的引导（rank guidance + manifold guidance）
        self.rank_net = RankNet(dim).to(self.config.device)
        self.vae = ManifoldVAE(dim, latent_dim=self.config.vae_latent_dim).to(self.config.device)
        self.synth = GPPosteriorMeanSampler(self.config)

        # ROOT Brownian Bridge 模型（与 ROOT 完全一致的训练 / 采样逻辑）
        self.bb_model = create_root_brownian_bridge(dim).to(self.config.device)
        self.bb_optimizer = torch.optim.Adam(self.bb_model.get_parameters(), lr=self.config.bridge_lr)

        # EMA（与 ROOT 一致，测试时用 EMA 权重，常提升 1～3 点）
        self.use_ema = getattr(self.config, "use_ema", True)
        self._ema: EMA | None = None
        self._global_step = 0
        if self.use_ema:
            self._ema = EMA(ema_decay=getattr(self.config, "ema_decay", 0.995))
            self._ema.register(self.bb_model)

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
        使用 ROOT 的 BrownianBridgeModel 在 GP 合成的对上训练。
        与 ROOT use_fixed_gp_posterior 一致：生成一次对（gp_seed）、val_frac 留出 10%、小批量 64、100 epoch。
        """
        device = self.config.device
        x = x.to(device)
        y = y.to(device)
        val_frac = getattr(self.config, "val_frac", 0.1)
        batch_size = getattr(self.config, "bridge_batch_size", 64)
        start_ema_step = getattr(self.config, "start_ema_step", 4000)
        update_ema_interval = getattr(self.config, "update_ema_interval", 8)

        self.bb_model.train()
        self._global_step = 0
        pairs = self.synth.generate(x, y)
        if not pairs:
            print("[Bridge][ROOT] WARNING: no synthetic pairs generated, bridge will be skipped.")
            return

        # 与 ROOT create_train_dataloader 一致：train = function_samples[int(len*val_frac):]，即后 90%
        train_start = int(len(pairs) * val_frac)
        train_pairs = pairs[train_start:] if train_start < len(pairs) else pairs
        dataset = PairsDataset(train_pairs)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=False,
        )

        for epoch in range(self.config.bridge_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                (x_high, y_high), (x_low, y_low) = batch
                x_high = x_high.to(device)
                y_high = y_high.to(device).float()
                x_low = x_low.to(device)
                y_low = y_low.to(device).float()
                if y_high.dim() == 1:
                    y_high = y_high.unsqueeze(-1)
                if y_low.dim() == 1:
                    y_low = y_low.unsqueeze(-1)

                # 与 ROOT BBDMRunner.loss_fn 对齐的 CFG 训练：
                # 当采样阶段会使用 cfg_scale / cfg_weight > 0 时，
                # 这里以 cfg_dropout 概率随机将 (y_high, y_low) 置零，构造无条件分布。
                cfg_prob = getattr(self.config, "cfg_dropout", 0.15)
                if cfg_prob > 0.0:
                    rand_mask = torch.rand_like(y_high)
                    mask = rand_mask <= cfg_prob
                    y_high = y_high.masked_fill(mask, 0.0)
                    y_low = y_low.masked_fill(mask, 0.0)

                loss, _ = self.bb_model(x_high, y_high, x_low, y_low)
                self.bb_optimizer.zero_grad()
                loss.backward()
                self.bb_optimizer.step()
                self._global_step += 1
                if self._ema is not None and self._global_step % update_ema_interval == 0:
                    with_decay = self._global_step >= start_ema_step
                    self._ema.update(self.bb_model, with_decay=with_decay)
                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % max(1, self.config.bridge_epochs // 5) == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                print(f"[Bridge][ROOT][epoch {epoch+1}/{self.config.bridge_epochs}] loss≈{avg_loss:.4f}")

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        self.train_ranknet(x, y)
        self.train_vae(x)
        self.train_bridge(x, y)

    def sample(
        self,
        x_low: torch.Tensor,
        y_low: torch.Tensor,
        high_cond_score: float | None = None,
        cfg_weight: float = 0.0,
        ) -> torch.Tensor:
        """
        手动实现 Brownian Bridge 反向采样循环，并在每一步注入：
        - 目标分外推（target extrapolation）
        - Rank 梯度引导（rank guidance）
        - VAE 流形密度引导（manifold guidance）

        参数：
            x_low, y_low: 作为起点的 low candidates（已在外部按 ROOT 的逻辑选好、且已归一化）。
            high_cond_score: 目标高分（在同一归一化空间下），通常为 oracle_y_max_normalized * alpha。
            cfg_weight: classifier-free guidance 的权重（如 -1.5），与 ROOT BBDMRunner 一致。
        """
        device = self.config.device
        self.bb_model.eval()

        x_low = x_low.to(device)
        y_low = y_low.to(device).unsqueeze(-1)

        # ---------------------------
        # 1. 目标分条件
        #    当已提供 high_cond_score 时直接使用（main.py 已完成 oracle * alpha 的计算）；
        #    仅在未提供时才用 offline max * target_extrap_factor 作为退化方案。
        # ---------------------------
        if high_cond_score is not None:
            target_val = high_cond_score
        else:
            factor = getattr(self.config, "target_extrap_factor", 1.0)
            target_val = y_low.max().item() * factor
        y_high = torch.full_like(y_low, target_val)

        if self._ema is not None:
            self._ema.apply_shadow(self.bb_model)

        print(
            f"[Sample][RGNB] low candidates: mean={y_low.mean().item():.4f}, "
            f"min={y_low.min().item():.4f}, max={y_low.max().item():.4f}; "
            f"target_y={target_val:.4f}, cfg_weight={cfg_weight:.2f}, "
            f"lambda_rank={self.config.lambda_rank:.4f}, lambda_manifold={self.config.lambda_manifold:.4f}"
        )

        # 简单版本的 extract，仿 ROOT model.utils.extract
        def _extract(buffer: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
            # buffer: (T,), t: (B,)
            out = buffer.gather(0, t)
            while out.dim() < len(x_shape):
                out = out.unsqueeze(-1)
            return out.expand(x_shape)

        steps = self.bb_model.steps  # 采样时间步（从 T-1 到 0）
        num_steps = len(steps)
        T = self.bb_model.num_timesteps

        # 引导只在后期施加：仅当进度 > guidance_start_ratio 时注入
        guidance_start_ratio = getattr(self.config, "guidance_start_ratio", 0.5)
        guidance_start_idx = int(num_steps * guidance_start_ratio)
        lam_rank = getattr(self.config, "lambda_rank", 0.0)
        lam_mani = getattr(self.config, "lambda_manifold", 0.0)
        use_guidance = (lam_rank > 0.0 or lam_mani > 0.0)

        x_t = x_low.clone()

        for i in range(num_steps):
            t_idx = steps[i].item()
            t = torch.full((x_t.shape[0],), t_idx, device=device, dtype=torch.long)

            # ------ Score 网络预测 + CFG ------
            eps_cond = self.bb_model.denoise_fn(x_t, t, y_low, y_high)
            zeros_y = torch.zeros_like(y_low)
            eps_uncond = self.bb_model.denoise_fn(x_t, t, zeros_y, zeros_y)
            objective_recon = (1.0 + cfg_weight) * eps_cond - cfg_weight * eps_uncond

            x0_recon = self.bb_model.predict_x0_from_objective(x_t, x_low, t, objective_recon)

            if t_idx == 0:
                x_next_mean = x0_recon
                sigma_t_scalar = 0.0
                noise_term = 0.0
            else:
                n_t_idx = steps[i + 1].item()
                n_t = torch.full((x_t.shape[0],), n_t_idx, device=device, dtype=torch.long)

                m_t = _extract(self.bb_model.m_t, t, x_t.shape)
                m_nt = _extract(self.bb_model.m_t, n_t, x_t.shape)
                var_t = _extract(self.bb_model.variance_t, t, x_t.shape)
                var_nt = _extract(self.bb_model.variance_t, n_t, x_t.shape)

                sigma2_t = (var_t - var_nt * (1.0 - m_t) ** 2 / (1.0 - m_nt) ** 2) * var_nt / var_t
                sigma_t = torch.sqrt(sigma2_t) * self.bb_model.eta
                sigma_t_scalar = sigma_t.mean().item()

                noise = torch.randn_like(x_t)
                x_tminus_mean = (
                    (1.0 - m_nt) * x0_recon
                    + m_nt * x_low
                    + torch.sqrt((var_nt - sigma2_t) / var_t)
                    * (x_t - (1.0 - m_t) * x0_recon - m_t * x_low)
                )
                x_next_mean = x_tminus_mean
                noise_term = sigma_t * noise

            # ------ RGNB 引导（仅在后期步骤施加） ------
            if use_guidance and i >= guidance_start_idx and t_idx > 0:
                # 时间衰减系数：从 0 线性增长到 1（越接近 t=0 引导越强）
                progress = (i - guidance_start_idx) / max(num_steps - guidance_start_idx, 1)

                x_for_g = x_next_mean.detach().clone().requires_grad_(True)

                # Rank 引导
                if lam_rank > 0.0:
                    rs = self.rank_net(x_for_g).sum()
                    g_rank = torch.autograd.grad(rs, x_for_g, retain_graph=True)[0]
                    g_rank = g_rank / (g_rank.norm(dim=-1, keepdim=True) + 1e-8)
                else:
                    g_rank = torch.zeros_like(x_next_mean)

                # Manifold 引导（确定性 VAE 路径，已在上面修复）
                if lam_mani > 0.0:
                    lp = self.vae.log_density_proxy(x_for_g).sum()
                    g_mani = torch.autograd.grad(lp, x_for_g)[0]
                    g_mani = g_mani / (g_mani.norm(dim=-1, keepdim=True) + 1e-8)
                else:
                    g_mani = torch.zeros_like(x_next_mean)

                raw_guidance = lam_rank * g_rank + lam_mani * g_mani

                # 梯度裁剪：引导幅度不超过当前步噪声尺度的一半
                if sigma_t_scalar > 0:
                    g_norm = raw_guidance.norm(dim=-1, keepdim=True)
                    max_norm = 0.5 * sigma_t_scalar
                    raw_guidance = raw_guidance * torch.clamp(max_norm / (g_norm + 1e-8), max=1.0)

                x_next_mean = x_next_mean + progress * raw_guidance

            # 合成噪声项
            if isinstance(noise_term, float) and noise_term == 0.0:
                x_t = x_next_mean
            else:
                x_t = x_next_mean + noise_term

        # 采样结束，恢复 EMA 权重，并返回与外部无梯度关联的结果
        if self._ema is not None:
            self._ema.restore(self.bb_model)

        return x_t.detach()
