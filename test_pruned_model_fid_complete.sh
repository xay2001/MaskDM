#!/bin/bash

# 完整流程：采样 + FID评估
# 测试剪枝后模型的生成质量
# 模型: weight_magnitude pruned (50% pruning ratio)
# 日期: 2025-10-19

set -e  # 遇到错误立即退出

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/pruned_model_fid_test_${TIMESTAMP}.log"

mkdir -p logs

echo "=========================================="
echo "剪枝模型完整FID测试流程"
echo "时间戳: ${TIMESTAMP}"
echo "日志保存: ${LOG_FILE}"
echo "=========================================="

# Step 1: 采样
echo ""
echo "Step 1/2: 从剪枝模型采样 50,000 张图像..."
echo ""
bash scripts/sampling/sample_pruned_weight_magnitude_cifar10.sh 2>&1 | tee -a ${LOG_FILE}

# Step 2: 计算FID
echo ""
echo "Step 2/2: 计算FID分数..."
echo ""
bash scripts/evaluation/fid_pruned_weight_magnitude_cifar10.sh 2>&1 | tee -a ${LOG_FILE}

echo ""
echo "=========================================="
echo "✅ 完整测试流程完成！"
echo "日志已保存到: ${LOG_FILE}"
echo "=========================================="

