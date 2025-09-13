#!/bin/bash

# 测试原始模型+初始mask的FID脚本
# 使用较小的样本数量进行快速测试

echo "开始测试原始模型+初始mask的FID..."

cd /data/xay/MaskDM

python test_original_model_with_initial_mask_fid.py \
    --original_model "/data/xay/MaskDM/pretrained/ddpm_ema_cifar10" \
    --total_samples 1000 \
    --batch_size 50 \
    --ddim_steps 50 \
    --seed 42 \
    --output_dir "/data/xay/MaskDM/samples_test_original_initial_mask" \
    --reference_stats "/data/xay/MaskDM/run/fid_stats_cifar10.npz" \
    --device "cuda"

echo "FID测试完成!"


