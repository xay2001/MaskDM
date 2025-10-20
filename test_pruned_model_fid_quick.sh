#!/bin/bash

# 快速FID测试（小样本量，用于快速验证）
# 模型: weight_magnitude pruned (50% pruning ratio)
# 样本数: 1000 (约2-3分钟)

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "剪枝模型快速FID测试"
echo "样本数: 1000 (快速验证)"
echo "时间戳: ${TIMESTAMP}"
echo "=========================================="

cd /data/xay/MaskDM

# Step 1: 快速采样 1000 张图像
echo ""
echo "Step 1/2: 采样 1000 张图像..."
echo ""

python ddpm_sample.py \
    --output_dir run/sample/pruned_weight_magnitude_cifar10_quick \
    --batch_size 100 \
    --model_path run/pruned/weight_magnitude/ddpm_cifar10_weight_pruned \
    --pruned_model_ckpt run/pruned/weight_magnitude/ddpm_cifar10_weight_pruned/pruned/unet_pruned.pth \
    --total_samples 1000 \
    --ddim_steps 100 \
    --skip_type uniform \
    --seed 42

echo ""
echo "Step 2/2: 计算FID分数..."
echo ""

# Step 2: 计算FID
python fid_score.py \
    run/sample/pruned_weight_magnitude_cifar10_quick/process_0 \
    run/fid_stats_cifar10.npz \
    --batch-size 50 \
    --device cuda

echo ""
echo "=========================================="
echo "✅ 快速测试完成！"
echo "注意: 这是小样本测试，FID可能不够准确"
echo "如需准确结果，请运行完整测试（50000样本）"
echo "=========================================="

