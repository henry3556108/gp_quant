#!/bin/bash
# 低 CPU 使用率版本的多樣性計算腳本

# 使用較少的工作進程（2 個而不是 4-8 個）
# 並在每個世代之間添加短暫休息

python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir "$1" \
    --n_workers 2 \
    "$@"
