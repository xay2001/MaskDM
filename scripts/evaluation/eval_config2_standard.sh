#!/bin/bash
# Config2 Standard 完整评估脚本 - 采样 + FID测试

set -e

MODEL_NAME="config2_standard"
MODEL_PATH="Maskpro/train_result/config2_standard/lr1.0_epoch2000_logits5.0_size20000_diffusion/checkpoint"
SAMPLE_DIR="run/sample/maskpro/${MODEL_NAME}"
REAL_STATS="run/stats/cifar10_real_stats.npz"
CIFAR10_REAL="run/real_images/cifar10_images"

echo "=========================================="
echo "Config2 Standard 完整评估流程"
echo "=========================================="

# # 创建必要目录
# mkdir -p run/sample/maskpro run/stats run/real_images

# echo "Step 1: 采样图像..."
# python ddpm_sample.py \
#     --output_dir ${SAMPLE_DIR} \
#     --batch_size 128 \
#     --model_path ${MODEL_PATH} \
#     --total_samples 50000 \
#     --ddim_steps 100 \
#     --skip_type uniform \
#     --seed 42

echo "Step 2: 准备真实CIFAR-10统计 (如果需要)..."
if [ ! -f ${REAL_STATS} ]; then
    echo "生成真实CIFAR-10统计..."
    python fid_score.py \
        --save-stats \
        --batch-size 128 \
        --num_samples 50000 \
        --res 32 \
        --dataset_name cifar10 \
        ${CIFAR10_REAL} \
        ${REAL_STATS}
else
    echo "真实CIFAR-10统计已存在"
fi

echo "Step 3: 计算FID分数..."
python fid_score.py \
    --batch-size 128 \
    --num_samples 50000 \
    --res 32 \
    --dataset_name cifar10 \
    ${REAL_STATS} \
    ${SAMPLE_DIR}

echo "=========================================="
echo "评估完成！"
echo "结果保存在: ${SAMPLE_DIR}"
echo "=========================================="
