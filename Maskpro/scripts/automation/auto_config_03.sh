#!/bin/bash
# 自动化配置: 0.3剪枝率实验
# Automation config for 0.3 pruning rate experiments

set -e

CONFIG_NAME="config_03_pruning"
GPU_ID="cuda:1"
BASE_DIR="/data/xay/MaskDM/Maskpro"

echo "=========================================="
echo "启动 ${CONFIG_NAME} 实验"
echo "GPU: ${GPU_ID}"
echo "0.3剪枝率 mask目录: initial_mask_diffusion_30"
echo "=========================================="

# 切换到正确目录
cd "${BASE_DIR}"

# 检查mask目录是否存在
if [ ! -d "initial_mask_diffusion_30" ]; then
    echo "❌ 错误: initial_mask_diffusion_30 目录不存在!"
    echo "请先运行 get_mask_diffusion.py 提取0.3剪枝率的mask"
    exit 1
fi

# 创建结果和日志目录
mkdir -p train_result/${CONFIG_NAME}
mkdir -p test_result/${CONFIG_NAME}
mkdir -p baseline_losses
mkdir -p learned_mask_diffusion_30

echo "Step 1/3: 计算基线损失..."
python inference_loss_diffusion.py \
    --original_model "../pretrained/ddpm_ema_cifar10" \
    --dataset cifar10 \
    --dataset_size 20000 \
    --batch_size 32 \
    --device ${GPU_ID} \
    --initial_mask_path "initial_mask_diffusion_30" \
    --learned_mask_path "learned_mask_diffusion_30" \
    --targets all

if [ $? -ne 0 ]; then
    echo "❌ 基线损失计算失败!"
    exit 1
fi

echo "✅ 基线损失计算完成"

echo "Step 2/3: 训练MaskPro模型..."
CUDA_VISIBLE_DEVICES=${GPU_ID##*:} python train_diffusion.py \
    --original_model "../pretrained/ddpm_ema_cifar10" \
    --dataset "cifar10" \
    --lr 1.0 \
    --epoch 2000 \
    --logits 5.0 \
    --dataset_size 20000 \
    --batch_size 32 \
    --max_step 15000 \
    --targets all \
    --save \
    --initial_mask_path "initial_mask_diffusion_30" \
    --learned_mask_path "learned_mask_diffusion_30" \
    --output_dir "train_result/${CONFIG_NAME}" \
    --project_name "DDPM-MaskPro-30" \
    --experiment_name "${CONFIG_NAME}_$(date +%m%d_%H%M)"

if [ $? -ne 0 ]; then
    echo "❌ 训练失败!"
    exit 1
fi

echo "✅ 训练完成"

# 查找训练结果目录
TRAIN_RESULT_DIR=$(find train_result/${CONFIG_NAME} -name "lr*_diffusion" -type d | head -1)

if [ -z "$TRAIN_RESULT_DIR" ]; then
    echo "❌ 未找到训练结果目录!"
    exit 1
fi

echo "Step 3/3: 测试评估模型..."
python test_ddpm_maskpro.py \
    --checkpoint_path "${TRAIN_RESULT_DIR}/checkpoint" \
    --original_model "../pretrained/ddpm_ema_cifar10" \
    --num_samples 1000 \
    --compute_fid \
    --output_dir "test_result/${CONFIG_NAME}"

if [ $? -ne 0 ]; then
    echo "❌ 测试失败!"
    exit 1
fi

echo "✅ 测试完成"

echo "=========================================="
echo "${CONFIG_NAME} 实验完成!"
echo "=========================================="
echo "训练结果: ${TRAIN_RESULT_DIR}"
echo "测试结果: test_result/${CONFIG_NAME}"
echo ""
echo "重要文件:"
echo "- 训练摘要: ${TRAIN_RESULT_DIR}/training_summary.json"
echo "- 模型检查点: ${TRAIN_RESULT_DIR}/checkpoint/"
echo "- 测试结果: test_result/${CONFIG_NAME}/corrected_evaluation_results.json"
echo ""
echo "0.3剪枝率实验全部完成! 🎉"
