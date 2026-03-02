#!/bin/bash
# RGNB Ant 任务训练与评测
cd "$(dirname "$0")/.."
python main.py -c configs/Ant.yaml --gpu_ids 0 --max_seeds 1
