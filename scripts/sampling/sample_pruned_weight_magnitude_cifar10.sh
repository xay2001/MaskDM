#!/bin/bash

# 测试剪枝后模型的采样脚本
# 模型: weight_magnitude pruned (50% pruning ratio)
# 日期: 2025-10-19

echo "=========================================="
echo "开始从剪枝模型采样图像..."
echo "模型: ddpm_cifar10_weight_pruned"
echo "剪枝方法: weight_magnitude"
echo "=========================================="

cd /data/xay/MaskDM

python ddpm_sample.py \
    --output_dir run/sample/pruned_weight_magnitude_cifar10_50percent \
    --batch_size 128 \
    --model_path run/pruned/weight_magnitude/ddpm_cifar10_weight_pruned \
    --pruned_model_ckpt run/pruned/weight_magnitude/ddpm_cifar10_weight_pruned/pruned/unet_pruned.pth \
    --total_samples 50000 \
    --ddim_steps 100 \
    --skip_type uniform \
    --seed 42

echo "=========================================="
echo "采样完成！"
echo "样本保存在: run/sample/pruned_weight_magnitude_cifar10_50percent/"
echo "=========================================="

