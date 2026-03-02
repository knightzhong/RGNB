"""RGNB 数据集加载模块，参考 ROOT 的 design_bench 集成。"""

from __future__ import annotations

import numpy as np
import torch

try:
    import design_bench
except ImportError:
    design_bench = None


# 各任务的 oracle 分数范围（用于归一化评估）
TASK_ORACLE = {
    "TFBind8-Exact-v0": {"min": 0.0, "max": 1.0, "best": 0.43929616},
    "TFBind10-Exact-v0": {"min": -1.8585268, "max": 2.1287067, "best": 0.005328223},
    "AntMorphology-Exact-v0": {"min": -386.90036, "max": 590.24445, "best": 165.32648},
    "DKittyMorphology-Exact-v0": {"min": -880.4585, "max": 340.90985, "best": 199.36252},
}


def load_design_bench_task(task_name: str, max_samples: int | None = None):
    """加载 design_bench 任务。"""
    if design_bench is None:
        raise ImportError("请安装 design_bench: pip install design-bench")
    kwargs = {}
    if task_name == "TFBind10-Exact-v0":
        kwargs["dataset_kwargs"] = {"max_samples": max_samples or 10000}
    task = design_bench.make(task_name, **kwargs)
    return task


def get_offline_data(
    task_name: str,
    normalize_x: bool = True,
    normalize_y: bool = True,
    seed: int = 0,
    max_samples: int | None = None,
):
    """
    从 design_bench 加载离线数据，并可选归一化。

    Returns:
        offline_x: (N, D) 设计向量
        mean_x, std_x: 用于反归一化
        offline_y: (N,) 分数
        mean_y, std_y: 用于反归一化
    """
    task = load_design_bench_task(task_name, max_samples)
    offline_x = task.x

    x_shape = None
    if task.is_discrete:
        offline_x = task.to_logits(offline_x)
        x_shape = offline_x.shape[1:]  # (d1, d2) 用于 reshape 回 predict 所需格式
        offline_x = offline_x.reshape(offline_x.shape[0], -1)

    mean_x = np.mean(offline_x, axis=0)
    std_x = np.std(offline_x, axis=0)
    std_x = np.where(std_x == 0, 1.0, std_x)

    offline_y = np.array(task.y).reshape(-1)
    mean_y = np.mean(offline_y)
    std_y = np.std(offline_y)
    if std_y == 0:
        std_y = 1.0

    np.random.seed(seed)
    shuffle_idx = np.random.permutation(offline_x.shape[0])
    offline_x = offline_x[shuffle_idx]
    offline_y = offline_y[shuffle_idx]
    offline_y = offline_y.reshape(-1)

    x_t = torch.from_numpy(offline_x).float()
    mean_x_t = torch.from_numpy(mean_x).float()
    std_x_t = torch.from_numpy(std_x).float()
    y_t = torch.from_numpy(offline_y).float()
    mean_y_t = torch.tensor(mean_y, dtype=torch.float32)
    std_y_t = torch.tensor(std_y, dtype=torch.float32)

    if normalize_x:
        x_t = (x_t - mean_x_t) / std_x_t
    if normalize_y:
        y_t = (y_t - mean_y_t) / std_y_t

    return {
        "x": x_t,
        "y": y_t,
        "mean_x": mean_x_t,
        "std_x": std_x_t,
        "mean_y": mean_y_t,
        "std_y": std_y_t,
        "task": task,
        "is_discrete": task.is_discrete,
        "x_shape": x_shape,
    }


def denormalize_x(x: torch.Tensor, mean_x: torch.Tensor, std_x: torch.Tensor) -> torch.Tensor:
    """将归一化后的设计反归一化。"""
    return x * std_x + mean_x


def evaluate_with_oracle(
    task,
    samples: torch.Tensor,
    mean_x: torch.Tensor,
    std_x: torch.Tensor,
    task_name: str,
    is_discrete: bool = False,
    x_shape=None,
) -> torch.Tensor:
    """
    使用 oracle 评估采样结果，返回真实分数。

    Args:
        task: design_bench task
        samples: 归一化后的采样 (N, D)
        mean_x, std_x: 归一化参数
        task_name: 任务名
        is_discrete: 是否离散任务
        x_shape: 离散任务时的原始 x 形状

    Returns:
        high_true_scores: (N,) 真实 oracle 分数
    """
    denorm = samples.cpu() * std_x.cpu() + mean_x.cpu()
    if is_discrete and x_shape is not None:
        denorm = denorm.reshape(denorm.shape[0], *x_shape)
    scores = task.predict(denorm.numpy())
    return torch.from_numpy(scores).float()


def compute_normalized_percentiles(
    scores: torch.Tensor,
    task_name: str,
) -> tuple[float, float, float]:
    """
    计算归一化后的 100th、80th、50th 分位数。

    Returns:
        (p100, p80, p50) 归一化到 [0, 1] 的分数
    """
    oracle = TASK_ORACLE.get(task_name, {"min": scores.min().item(), "max": scores.max().item()})
    y_min = oracle["min"]
    y_max = oracle["max"]
    normalized = (scores - y_min) / (y_max - y_min)
    percentiles = torch.quantile(normalized, torch.tensor([1.0, 0.8, 0.5]), interpolation="higher")
    return percentiles[0].item(), percentiles[1].item(), percentiles[2].item()
