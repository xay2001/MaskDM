#!/bin/bash
# FID evaluation for Config2 Standard model

set -e

MODEL_NAME="config2_standard"
SAMPLE_DIR="run/sample/maskpro/${MODEL_NAME}"
REAL_STATS="run/stats/cifar10_real_stats.npz"
CIFAR10_REAL="run/real_images/cifar10"

echo "=========================================="
echo "FID Evaluation for ${MODEL_NAME}"
echo "=========================================="

# 创建必要目录
mkdir -p run/stats run/real_images

# Step 1: 如果需要，先生成真实CIFAR-10统计
if [ ! -f ${REAL_STATS} ]; then
    echo "Generating real CIFAR-10 statistics..."
    python fid_score.py \
        --save-stats \
        --batch-size 128 \
        --num_samples 50000 \
        --res 32 \
        --dataset_name cifar10 \
        ${CIFAR10_REAL} \
        ${REAL_STATS}
else
    echo "Real CIFAR-10 statistics already exist."
fi

# Step 2: 计算FID分数
echo "Computing FID score for ${MODEL_NAME}..."
python fid_score.py \
    --batch-size 128 \
    --num_samples 50000 \
    --res 32 \
    --dataset_name cifar10 \
    ${REAL_STATS} \
    ${SAMPLE_DIR}
