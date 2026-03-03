# RGNB DKitty 与 ROOT 对齐要点总结

基于 `dkitty.log` 复现效果（100th≈0.97）的关键实现与配置如下。

---

## 1. 固定 GP 模式（use_fixed_gp_posterior）

- **GP 只生成一次对**：训练前调用一次 `synth.generate(x, y)`，得到固定的一组 (x_low, x_high) 对，整个训练过程复用，与 ROOT 的 `use_fixed_gp_posterior` 一致。
- **固定随机种子**：在 `generate()` 开头 `torch.manual_seed(gp_seed)`（默认 0），与 ROOT `sampling_data_from_fixed_GP(..., seed=0)` 一致，保证可复现、且与 ROOT 可比。

---

## 2. 训练集划分（val_frac）——只影响「桥」的训练样本

- **只用 90% 的 pairs 训练桥**：`train_pairs = pairs[int(len(pairs)*val_frac):]`（val_frac=0.1），与 ROOT 的 `create_train_dataloader` 一致（后 90% 为 train，前 10% 为 val）。  
- 这 10% 在 RGNB 里未参与验证；在 ROOT 里用作验证集（validation）和 LR 调度。

---

## 3. 小批量 + 100 epoch

- **DataLoader**：用 `PairsDataset` + `DataLoader`，`batch_size=64`、`shuffle=True`，按 batch 更新，不再每 epoch 全量一步更新。
- **epoch 数**：`bridge_epochs: 100`，与 ROOT `n_epochs: 100` 一致。

---

## 4. 模型初始化（weights_init）

- 在 `root_bb.create_root_brownian_bridge()` 中，创建 BrownianBridge 后执行 `model.apply(weights_init)`，使用 ROOT 的 `runners.utils.weights_init`，与 BBDMRunner 一致。

---

## 5. EMA（效果关键）

- **训练**：每 `update_ema_interval=8` 步更新一次 EMA，且仅当 `global_step >= start_ema_step=4000` 时使用 decay（0.995）。
- **测试**：采样前调用 `ema.apply_shadow(bb_model)`，用 EMA 权重做推理，与 ROOT 测试前 `apply_ema()` 一致。
- 配置：`use_ema: true`，`ema_decay: 0.995`，`start_ema_step: 4000`，`update_ema_interval: 8`。

---

## 6. 评测与采样

- **起点**：`type_sampling: highest`，取 offline 中 y 最高的 128 个点作为 low candidates，与 ROOT `sampling_from_offline_data(..., type='highest')` 一致。
- **目标分数**：`high_cond_score = normalized_oracle_y_max * alpha`（alpha=0.8），与 BaseRunner.test 一致。
- **归一化指标**：`(score - oracle_min) / (oracle_max - oracle_min)`，再算 100th/80th/50th 分位数，与 ROOT 一致。

---

## 配置速查（configs/Dkitty.yaml）

| 项 | 值 | 说明 |
|----|-----|------|
| bridge_epochs | 100 | 与 ROOT n_epochs 一致 |
| val_frac | 0.1 | 训练用 90% 对 |
| bridge_batch_size | 64 | 小批量 SGD |
| gp_seed | 0 | 固定 GP 生成，可复现 |
| use_ema | true | 测试用 EMA 权重 |
| ema_decay | 0.995 | 与 ROOT 一致 |
| start_ema_step | 4000 | 再开始带 decay 的 EMA |
| update_ema_interval | 8 | 每 8 步更新 EMA |

---

## 典型结果（dkitty.log）

- **100th**: 0.9730  
- **80th**: 0.9356  
- **50th**: 0.9138  

---

## GP 的“训练样本”是否与 ROOT 统一？

GP 涉及两类数据，都和 ROOT 对齐了：

| 用途 | RGNB | ROOT (use_fixed_gp_posterior) |
|------|------|--------------------------------|
| **拟合 GP 用的数据** | 传入的 `(x, y)` 全量（当 `n ≤ gp_max_fit` 时），即 `get_offline_data` 得到的整份离线数据 | `offline_x`, `offline_y` 全量（DKitty 等非 TFBind8） |
| **生成 (x_low, x_high) 对的起点** | `x_pool` = 全量中 **y 最高的 1024 个点**（`type_of_initial_points: highest`） | `best_x` = `offline_x[argsort(offline_y)[-1024:]]`，同样是 y 最高的 1024 个点 |

- **拟合 GP**：两边都是用「全量离线 (x,y)」拟合一次，逻辑一致。DKitty 配置里 `gp_max_fit: 100000`，一般不小于数据量，故不会触发随机子采样；若数据量 &gt; 100000，RGNB 会子采样，此时与 ROOT 不全一致，可把 `gp_max_fit` 调大以保持全量。
- **起点池**：都是「y 最大的 1024 个点」作为梯度上升/下降的起点，与 ROOT 的 `sampling_data_from_fixed_GP(x_train=best_x, ...)` 一致。

因此：**桥的训练样本** = 上述 GP 生成的全部 pairs 里取后 90%；**GP 的“训练样本”**（拟合用全量、起点用 top-1024）已与 ROOT 统一。

若仍与 ROOT/ECG 有细小差异，可再核对：离线数据 shuffle 是否用同一 seed、是否启用 CFG（`use_classifier_free_guidance: true` + `classifier_free_guidance_weight: -1.5`）等。
