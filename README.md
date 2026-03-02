# RGNB

Rank-Guided Neural Bridge (RGNB) 的 PyTorch 实现，用于**离线黑盒优化**。

## 已实现内容

- **Algorithm 1 合成数据生成**
  - 使用 `gpytorch` 的 RBF GP 拟合离线数据。
  - 每轮随机采样长度尺度 `l ~ U(l0-delta, l0+delta)`。
  - 从离线数据采样 `n_p` 起点，分别执行 `M` 步梯度上升和下降，构造 `(x_T, y_T, x_0, y_0)` 配对数据。
- **RankNet**
  - 3 层 MLP 输出标量排序分数。
  - 提供 `pairwise BCE` 和 `pairwise hinge` 两种损失。
- **ScoreNetwork**
  - 4 层 MLP + Swish 激活。
  - 输入 `(x_t, t, y_cond)`，用于布朗桥噪声预测。
- **Manifold VAE**
  - 轻量级 VAE 学习离线分布，提供 `log p(x)` 代理梯度用于流形约束。
- **概率桥训练（Algorithm 2）**
  - 使用布朗桥采样公式构建 `x_t`。
  - 优化噪声预测 MSE（对应论文公式）。
  - Adam + `lr=1e-3`，默认训练 100 epoch。
- **闭环引导推理**
  - 从离线数据 top-128 点开始反向去噪。
  - 同时使用：
    - classifier-free guidance（条件/无条件融合）；
    - Rank 梯度引导；
    - 流形密度梯度引导。
  - 采样循环中保持梯度路径可追踪（`create_graph=True`）。

## 安装

```bash
pip install -r requirements.txt
# 或
pip install torch gpytorch design-bench pyyaml tqdm
```

### design_bench 离线数据（Morphology 任务需额外安装）

运行 Ant / DKitty 等 Morphology 任务前，需下载 design_bench 离线数据（参考 ROOT 项目）：

```bash
pip install gdown
gdown 'https://drive.google.com/uc?id=1n5R0p_7OAejDts6B_WH6qbBRfT8BEiiN'
unzip design_bench_data.zip
mv -v design_bench_data $CONDA_PREFIX/lib/python3.x/site-packages  # 替换 3.x 为实际版本
```

## 训练与评测（design_bench）

参考 ROOT 项目，支持完整训练流程与指标评测：

```bash
# Ant 任务（默认 8 个 seed）
python main.py -c configs/Ant.yaml --gpu_ids 0

# DKitty 任务
python main.py -c configs/Dkitty.yaml --gpu_ids 0

# 仅评测已训练模型
python main.py -c configs/Ant.yaml --resume results/AntMorphology_Exact_v0/seed0/rgnb_model.pt --eval_only
```

评测指标：Normalized 100th / 80th / 50th percentile score（与 ROOT 一致）。

## 快速使用（API）

```python
import torch
from rgnb import RGNBModel, RGNBConfig

# 假设已有离线数据
x = torch.randn(2000, 16)
y = -(x**2).sum(dim=-1) + 0.1 * torch.randn(2000)

cfg = RGNBConfig(device="cpu")
model = RGNBModel(dim=x.size(1), config=cfg)
model.fit(x, y)

x_opt = model.sample(x, y)
print(x_opt.shape)  # [top_k, dim]
```
