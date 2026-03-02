from __future__ import annotations

"""
将 ROOT 项目的 BrownianBridgeModel 引入到 RGNB 中使用。

用法示例（在 RGNB 内部）：

    from rgnb.root_bb import create_root_brownian_bridge
    bb_model = create_root_brownian_bridge(dim)
    loss, log = bb_model(x_high, y_high, x_low, y_low)
    x_samples = bb_model.sample(x_low, y_low, y_high)

注意：这里只负责把 ROOT 的 BrownianBridgeModel 作为一个可用模块引入，
并构造一个最小可用的 model_config；如何与 RGNB 当前 pipeline 完整对接，
可以在 RGNBModel 内部逐步替换自定义 BrownianBridgeTrainer。
"""

import os
import sys
import argparse
from typing import Any


def _ensure_root_on_path() -> None:
    """把 workforMBO/ROOT/ROOT 加到 sys.path，方便直接 import ROOT 里的模块。"""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    root_root = os.path.join(this_dir, "..", "..", "ROOT", "ROOT")
    root_root = os.path.normpath(root_root)
    if os.path.isdir(root_root) and root_root not in sys.path:
        sys.path.append(root_root)


_ensure_root_on_path()

try:
    # ROOT 项目中的 BrownianBridgeModel
    from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel  # type: ignore
except Exception as e:  # pragma: no cover - 环境缺失时的友好错误
    BrownianBridgeModel = None  # type: ignore
    _import_error = e
else:
    _import_error = None


class DotNamespace(argparse.Namespace):
    """简单的点访问 Namespace，用来模拟 ROOT 中的 OmegaConf config 结构。"""

    def __getattr__(self, item: str) -> Any:
        if item in self.__dict__:
            return self.__dict__[item]
        raise AttributeError(item)

    def __contains__(self, item: str) -> bool:
        return item in self.__dict__


def build_minimal_bb_config(dim: int) -> DotNamespace:
    """
    构造一个最小可用的 BrownianBridgeModel 配置，直接在 design 向量空间上工作。

    这里参考 ROOT 的 configs/Dkitty_FlowMatching.yaml 等设置，
    但去掉与图像相关的部分，只保留 MLP 所需的 image_size/hidden_size。
    """
    cfg = DotNamespace()

    # BB 节点
    cfg.BB = DotNamespace()
    params = DotNamespace()

    # 时间与方差 schedule
    params.num_timesteps = 200
    params.mt_type = "linear"
    params.max_var = 1.0
    params.eta = 1.0

    # 采样相关
    params.skip_sample = False
    params.sample_type = "linear"
    params.sample_step = 200

    # 损失与目标
    params.loss_type = "l2"
    params.objective = "grad"

    # MLP 参数：直接在 design 向量空间上工作
    mlp_params = DotNamespace()
    mlp_params.image_size = dim
    mlp_params.hidden_size = 1024
    mlp_params.condition_key = "SpatialRescaler"
    params.MLPParams = mlp_params

    cfg.BB.params = params

    # 其余字段在当前集成中暂不需要
    return cfg


def create_root_brownian_bridge(dim: int) -> BrownianBridgeModel:
    """
    创建一个在 design 向量空间上工作的 ROOT BrownianBridgeModel 实例。

    Args:
        dim: 设计向量维度
    """
    if BrownianBridgeModel is None:
        raise ImportError(
            "无法导入 ROOT 的 BrownianBridgeModel，请确认 workforMBO/ROOT/ROOT 在本仓库旁边，"
            f"以及导入错误为：{_import_error}"
        )
    cfg = build_minimal_bb_config(dim)
    return BrownianBridgeModel(cfg)

