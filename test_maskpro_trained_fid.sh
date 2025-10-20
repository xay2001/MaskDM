#!/bin/bash

# 测试 MaskPro 训练后模型的 FID (包含稀疏性)
# 模型稀疏度: 45.95%
# 样本数: 1000 (快速测试)

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_PATH="/data/xay/MaskDM/Maskpro/train_result/cifar10_bs64_lr1_logit5_1019/lr1.0_epoch100_logits5.0_size20000_diffusion/checkpoint"
OUTPUT_DIR="run/sample/maskpro_trained_sparse_${TIMESTAMP}"

echo "=========================================="
echo "MaskPro训练后模型FID测试 (稀疏模型)"
echo "模型稀疏度: 45.95%"
echo "样本数: 1000"
echo "=========================================="

cd /data/xay/MaskDM

# Step 1: 采样
echo ""
echo "Step 1/2: 采样 1000 张图像..."
python ddpm_sample.py \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 100 \
    --model_path ${MODEL_PATH} \
    --total_samples 1000 \
    --ddim_steps 100 \
    --skip_type uniform \
    --seed 42

# Step 2: 计算FID
echo ""
echo "Step 2/2: 计算FID分数..."
python fid_score.py \
    ${OUTPUT_DIR}/process_0 \
    run/fid_stats_cifar10.npz \
    --batch-size 50 \
    --device cuda

echo ""
echo "=========================================="
echo "✅ 测试完成！"
echo "样本保存在: ${OUTPUT_DIR}"
echo "模型稀疏度: 45.95% (真正的稀疏模型)"
echo "=========================================="

