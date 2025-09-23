#!/bin/bash

# A100上的2:4剪枝模型推理速度测试脚本

echo "🚀 开始A100上的推理速度对比测试..."
echo "📊 测试Dense模型 vs 2:4剪枝模型"

cd /data/xay/MaskDM

# 激活conda环境并运行测试
conda activate prunedm

echo "🔧 检查CUDA和GPU状态..."
python -c "
import torch
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'当前GPU: {torch.cuda.get_device_name()}')
    print(f'CUDA版本: {torch.version.cuda}')
"

echo "⚡ 开始性能基准测试..."
python speed_benchmark.py

echo "✅ 测试完成！"
