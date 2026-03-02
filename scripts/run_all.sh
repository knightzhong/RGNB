#!/bin/bash
# RGNB 全任务训练与评测（参考 ROOT scripts/bash.sh）
cd "$(dirname "$0")/.."
python main.py -c configs/Ant.yaml --gpu_ids 0
python main.py -c configs/Dkitty.yaml --gpu_ids 0
python main.py -c configs/TfBind8.yaml --gpu_ids 0
python main.py -c configs/TfBind10.yaml --gpu_ids 0
