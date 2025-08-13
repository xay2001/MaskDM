#!/bin/bash

# 精确50%剪枝脚本
# 通过迭代调整找到最接近50%的剪枝率

echo "🎯 目标：精确50%参数剪枝"

# 尝试不同的剪枝率，找到最接近50%的结果
for ratio in 0.20 0.22 0.24 0.26 0.28; do
    echo "尝试剪枝率: $ratio"
    
    python ddpm_prune.py \
        --dataset cifar10 \
        --model_path pretrained/ddpm_ema_cifar10 \
        --save_path run/pruned/magnitude/test_${ratio} \
        --pruning_ratio $ratio \
        --batch_size 128 \
        --pruner magnitude \
        --device cuda:2 \
        > pruning_${ratio}.log 2>&1
    
    # 提取参数信息
    actual_reduction=$(grep "#Params:" pruning_${ratio}.log | tail -1)
    echo "结果: $actual_reduction"
    echo "---"
done

echo "请检查日志文件选择最接近50%的配置"