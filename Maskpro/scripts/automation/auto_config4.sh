#!/bin/bash
# 自动化配置4: GPU优化配置 - GPU 3
# Auto Configuration 4: GPU optimized setup on GPU 3

set -e  # 遇到错误立即退出

CONFIG_NAME="config4_efficient"
GPU_ID=3
BASE_DIR="/data/xay/MaskDM/Maskpro"

echo "=========================================="
echo "开始执行自动化配置4 - GPU优化配置"
echo "GPU: cuda:${GPU_ID}"
echo "配置名称: ${CONFIG_NAME}"
echo "=========================================="

# 创建结果目录
mkdir -p train_result test_result

echo "[$(date)] 步骤1: 生成基线损失..."
cd "${BASE_DIR}"
python inference_loss_diffusion.py \
      --original_model "../pretrained/ddpm_ema_cifar10" \
      --dataset "cifar10" \
      --dataset_size 25000 \
      --batch_size 128 \
      --max_batches 196 \
      --device "cuda:${GPU_ID}" \
      --targets all

if [ $? -eq 0 ]; then
    echo "[$(date)] 损失生成完成 ✓"
else
    echo "[$(date)] 损失生成失败 ✗" 
    exit 1
fi

echo "[$(date)] 步骤2: 开始训练..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_diffusion.py \
      --original_model "../pretrained/ddpm_ema_cifar10" \
      --dataset "cifar10" \
      --lr 1.5 \
      --epoch 1500 \
      --logits 6.0 \
      --dataset_size 25000 \
      --batch_size 128 \
      --max_step 12000 \
      --targets all \
      --save \
      --output_dir "train_result/${CONFIG_NAME}"

if [ $? -eq 0 ]; then
    echo "[$(date)] 训练完成 ✓"
else
    echo "[$(date)] 训练失败 ✗"
    exit 1
fi

echo "[$(date)] 步骤3: 开始测试..."
python test_ddpm_maskpro.py \
    --checkpoint_path "train_result/${CONFIG_NAME}" \
    --original_model ../pretrained/ddpm_ema_cifar10 \
    --num_inference_steps 150 \
    --num_samples 1500 \
    --batch_size 50 \
    --compute_fid \
    --output_dir "test_result/${CONFIG_NAME}"

if [ $? -eq 0 ]; then
    echo "[$(date)] 测试完成 ✓"
else
    echo "[$(date)] 测试失败 ✗"
    exit 1
fi

echo "=========================================="
echo "[$(date)] 配置4自动化流程完成！"
echo "训练结果: train_result/${CONFIG_NAME}"
echo "测试结果: test_result/${CONFIG_NAME}"
echo "=========================================="
