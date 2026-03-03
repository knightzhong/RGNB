#!/usr/bin/env python3
"""
RGNB 训练与评测入口。

参考 ROOT 项目结构，支持：
- design_bench 数据集加载
- YAML 配置
- 多 seed 重复实验
- 100th/80th/50th 分位数指标
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from rgnb import RGNBModel, RGNBConfig
from rgnb.data import (
    get_offline_data,
    evaluate_with_oracle,
    compute_normalized_percentiles,
    TASK_ORACLE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="RGNB: Rank-Guided Neural Bridge 训练与评测")
    parser.add_argument("-c", "--config", type=str, default="configs/Ant.yaml", help="配置文件路径")
    parser.add_argument("-s", "--seed", type=int, default=1234, help="随机种子")
    parser.add_argument("-r", "--result_path", type=str, default="results", help="结果保存目录")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU ID，-1 表示 CPU")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复")
    parser.add_argument("--max_seeds", type=int, default=8, help="重复实验的 seed 数量")
    parser.add_argument("--eval_only", action="store_true", help="仅评测（需配合 --resume 指定 checkpoint）")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载 YAML 配置。"""
    path = Path(config_path)
    if not path.is_absolute() and not path.exists():
        # 尝试相对于项目根目录
        script_dir = Path(__file__).resolve().parent
        alt_path = script_dir / config_path
        if alt_path.exists():
            path = alt_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_to_rgnb_config(cfg: dict, device: str) -> RGNBConfig:
    """将 YAML 配置转为 RGNBConfig。"""
    rgnb = cfg.get("rgnb", {})
    training = cfg.get("training", {})
    return RGNBConfig(
        ne=rgnb.get("ne", 8),
        np=rgnb.get("np", 1024),
        m_steps=rgnb.get("m_steps", 100),
        eta=rgnb.get("eta", 5e-3),
        l0=rgnb.get("l0", 1.0),
        delta=rgnb.get("delta", 0.5),
        gp_noise=rgnb.get("gp_noise", 1e-3),
        gp_max_fit=rgnb.get("gp_max_fit", 4096),
        gp_threshold_diff=rgnb.get("gp_threshold_diff", 1e-3),
        # 对齐 ROOT: type_of_initial_points（highest / lowest / all）
        gp_initial_points_type=rgnb.get("type_of_initial_points", rgnb.get("gp_initial_points_type", "all")),
        bridge_T=rgnb.get("bridge_T", 200),
        bridge_epochs=rgnb.get("bridge_epochs", 100),
        bridge_lr=rgnb.get("bridge_lr", 1e-3),
        top_k=rgnb.get("top_k", 128),
        cfg_scale=rgnb.get("cfg_scale", 2.0),
        lambda_rank=rgnb.get("lambda_rank", 0.0),
        lambda_manifold=rgnb.get("lambda_manifold", 0.0),
        cfg_dropout=rgnb.get(
            "cfg_dropout",
            training.get("classifier_free_guidance_prob", 0.15),
        ),
        target_extrap_factor=rgnb.get("target_extrap_factor", 1.0),
        guidance_start_ratio=rgnb.get("guidance_start_ratio", 0.5),
        vae_latent_dim=rgnb.get("vae_latent_dim", 16),
        gp_seed=rgnb.get("gp_seed", 0),
        val_frac=rgnb.get("val_frac", 0.1),
        bridge_batch_size=rgnb.get("bridge_batch_size", 64),
        use_ema=rgnb.get("use_ema", True),
        ema_decay=rgnb.get("ema_decay", 0.995),
        start_ema_step=rgnb.get("start_ema_step", 4000),
        update_ema_interval=rgnb.get("update_ema_interval", 8),
        device=device,
    )


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(config: dict, args) -> RGNBModel:
    """训练 RGNB 模型。"""
    set_random_seed(args.seed)
    device = "cpu" if args.gpu_ids == "-1" else f"cuda:{args.gpu_ids}"

    task_name = config["task"]["name"]
    normalize_x = config["task"].get("normalize_x", True)
    normalize_y = config["task"].get("normalize_y", True)

    print("=" * 60)
    print("[RGNB] 开始训练")
    print(f"  任务: {task_name}")
    print(f"  设备: {device}")
    print("=" * 60)

    # 加载数据
    max_samples = 10000 if task_name == "TFBind10-Exact-v0" else None
    data = get_offline_data(
        task_name=task_name,
        normalize_x=normalize_x,
        normalize_y=normalize_y,
        seed=args.seed,
        max_samples=max_samples,
    )
    x = data["x"].to(device)
    y = data["y"].to(device)

    # 初始化模型
    rgnb_cfg = config_to_rgnb_config(config, device)
    dim = x.size(1)
    model = RGNBModel(dim=dim, config=rgnb_cfg)

    # 从 checkpoint 恢复（可选）
    # if args.resume and os.path.isfile(args.resume):
    #     ckpt = torch.load(args.resume, map_location=device)
    #     model.rank_net.load_state_dict(ckpt["rank_net"])
    #     model.score_net.load_state_dict(ckpt["score_net"])
    #     model.vae.load_state_dict(ckpt["vae"])
    #     run_config = ckpt.get("config", config)
    #     print(f"[RGNB] 已从 {args.resume} 恢复")
    #     if args.eval_only:
    #         return model, run_config

    # 训练参数
    training_cfg = config.get("training", {})
    ranknet_epochs = training_cfg.get("ranknet_epochs", 50)
    vae_epochs = training_cfg.get("vae_epochs", 50)

    # 训练 RankNet、VAE 与 Brownian Bridge
    print("[RGNB] 训练 RankNet...")
    model.train_ranknet(x, y, epochs=ranknet_epochs)
    print("[RGNB] 训练 VAE...")
    model.train_vae(x, epochs=vae_epochs)
    print("[RGNB] 训练 Brownian Bridge (ROOT BrownianBridgeModel)...")
    model.train_bridge(x, y)

    # 保存 checkpoint
    # if args.result_path:
    #     save_dir = Path(args.result_path) / task_name.replace("-", "_") / f"seed{args.seed}"
    #     save_dir.mkdir(parents=True, exist_ok=True)
    #     ckpt_path = save_dir / "rgnb_model.pt"
    #     torch.save(
    #         {
    #             "rank_net": model.rank_net.state_dict(),
    #             "score_net": model.score_net.state_dict(),
    #             "vae": model.vae.state_dict(),
    #             "config": config,
    #         },
    #         ckpt_path,
    #     )
    #     print(f"[RGNB] 模型已保存至 {ckpt_path}")

    return model, config


def test(model: RGNBModel, config: dict, args, task) -> tuple[float, float, float]:
    """评测 RGNB 模型，返回 (p100, p80, p50)。"""
    set_random_seed(args.seed)
    device = "cpu" if args.gpu_ids == "-1" else f"cuda:{args.gpu_ids}"
    task_name = config["task"]["name"]

    # 加载数据（与训练时相同归一化）
    max_samples = 10000 if task_name == "TFBind10-Exact-v0" else None
    data = get_offline_data(
        task_name=task_name,
        normalize_x=config["task"].get("normalize_x", True),
        normalize_y=config["task"].get("normalize_y", True),
        seed=args.seed,
        max_samples=max_samples,
    )
    x = data["x"].to(device)
    y = data["y"].to(device)
    mean_x = data["mean_x"]
    std_x = data["std_x"]
    is_discrete = data["is_discrete"]
    x_shape = data.get("x_shape")

    # ===== 起点选择与目标高分，对齐 ROOT BaseRunner.test =====
    testing_cfg = config.get("testing", {})
    training_cfg = config.get("training", {})
    num_candidates = testing_cfg.get("num_candidates", 128)
    type_sampling = testing_cfg.get("type_sampling", "highest")
    alpha = testing_cfg.get("alpha", 0.8)

    # 选 low candidates（在 type_sampling='highest' 时即为 top-k high candidates）
    y_flat = y.view(-1)
    idx_sorted = torch.argsort(y_flat)
    if type_sampling == "highest":
        low_idx = idx_sorted[-num_candidates:]
    elif type_sampling == "lowest":
        low_idx = idx_sorted[:num_candidates]
    else:
        # 其它类型简单退化为 highest
        low_idx = idx_sorted[-num_candidates:]

    x_low = x[low_idx]
    y_low = y[low_idx]

    # oracle y_max -> 归一化到当前 offline 分布，再乘 alpha
    oracle = TASK_ORACLE.get(task_name, None)
    if oracle is not None:
        oracle_y_max = oracle["max"]
        norm_oracle_y_max = (oracle_y_max - data["mean_y"].item()) / data["std_y"].item()
        high_cond_score = norm_oracle_y_max * alpha
    else:
        high_cond_score = y.max().item()

    # 与 ROOT BBDMRunner.sample 一致：只有在 use_classifier_free_guidance=True 时才启用 CFG
    use_cfg_test = testing_cfg.get(
        "use_classifier_free_guidance",
        training_cfg.get("use_classifier_free_guidance", True),
    )
    cfg_weight = (
        testing_cfg.get("classifier_free_guidance_weight", 0.0) if use_cfg_test else 0.0
    )

    # 采样（由 ROOT BrownianBridgeModel + RGNB 引导联合完成）
    samples = model.sample(x_low, y_low, high_cond_score=high_cond_score, cfg_weight=cfg_weight)

    # 反归一化并评估
    scores = evaluate_with_oracle(
        task=task,
        samples=samples,
        mean_x=mean_x,
        std_x=std_x,
        task_name=task_name,
        is_discrete=is_discrete,
        x_shape=x_shape,
    )
    p100, p80, p50 = compute_normalized_percentiles(scores, task_name)
    return p100, p80, p50


def main():
    args = parse_args()

    if args.eval_only and not args.resume:
        raise ValueError("--eval_only 需配合 --resume 指定 checkpoint 路径")

    config = load_config(args.config)

    # eval_only 时从 checkpoint 读取 config，确保 task 一致
    if args.eval_only and args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        config = ckpt.get("config", config)

    task_name = config["task"]["name"]
    max_samples = 10000 if task_name == "TFBind10-Exact-v0" else None

    # 加载 task（用于评测）
    import design_bench
    if task_name != "TFBind10-Exact-v0":
        task = design_bench.make(task_name)
    else:
        task = design_bench.make(task_name, dataset_kwargs={"max_samples": max_samples or 10000})
    if task.is_discrete:
        task.map_to_logits()

    results_100, results_80, results_50 = [], [], []
    max_seeds = 1 #if args.eval_only else min(args.max_seeds, 8)

    for seed in range(max_seeds):
        args.seed = seed

        # 训练（或从 checkpoint 加载）
        model, run_config = train(config, args)

        # 评测
        p100, p80, p50 = test(model, run_config, args, task)
        print(f"Seed {seed} 结果: 100th={p100:.4f}, 80th={p80:.4f}, 50th={p50:.4f}")
        results_100.append(p100)
        results_80.append(p80)
        results_50.append(p50)

    results_100 = np.array(results_100)
    results_80 = np.array(results_80)
    results_50 = np.array(results_50)

    print("\n" + "=" * 60)
    print("RGNB 评测汇总")
    print("=" * 60)
    print("Normalized 100th percentile score:")
    print(f"  Mean: {np.mean(results_100):.4f}, Std: {np.std(results_100):.4f}")
    print("Normalized 80th percentile score:")
    print(f"  Mean: {np.mean(results_80):.4f}, Std: {np.std(results_80):.4f}")
    print("Normalized 50th percentile score:")
    print(f"  Mean: {np.mean(results_50):.4f}, Std: {np.std(results_50):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
