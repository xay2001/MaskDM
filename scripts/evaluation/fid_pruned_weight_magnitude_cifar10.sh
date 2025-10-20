#!/bin/bash

# 计算剪枝模型的FID分数
# 模型: weight_magnitude pruned (50% pruning ratio)
# 日期: 2025-10-19

echo "=========================================="
echo "开始计算FID分数..."
echo "生成图像: run/sample/pruned_weight_magnitude_cifar10_50percent/"
echo "参考统计: run/fid_stats_cifar10.npz"
echo "=========================================="

cd /data/xay/MaskDM

python fid_score.py \
    run/sample/pruned_weight_magnitude_cifar10_50percent/process_0 \
    run/fid_stats_cifar10.npz \
    --batch-size 50 \
    --device cuda

echo "=========================================="
echo "FID计算完成！"
echo "=========================================="

