#!/bin/bash

# 完整版mask约束微调脚本 - 完全匹配magnitude参数

echo "🚀 开始完整版mask约束微调（匹配magnitude设置）..."

# 设置路径
PRUNED_MODEL_PATH="/data/xay/MaskDM/Maskpro/train_result/config2_standard/lr1.0_epoch2000_logits5.0_size20000_diffusion/checkpoint"
OUTPUT_DIR="/data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2"

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "📁 剪枝模型路径: $PRUNED_MODEL_PATH"
echo "📁 输出目录: $OUTPUT_DIR"

# 完全匹配magnitude脚本的参数
python ddpm_train_simple_masked.py \
    --dataset cifar10 \
    --model_path $PRUNED_MODEL_PATH \
    --resolution 32 \
    --output_dir $OUTPUT_DIR \
    --train_batch_size 128 \
    --num_iters 100000 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --lr_warmup_steps 0 \
    --save_model_steps 1000 \
    --dataloader_num_workers 8 \
    --adam_weight_decay 0.00 \
    --ema_max_decay 0.9999 \
    --dropout 0.1 \
    --use_ema \
    --logger wandb \
    --overwrite_output_dir

echo "✅ 完整微调完成！"
echo "📊 输出目录: $OUTPUT_DIR"
echo "🖼️  生成样本: $OUTPUT_DIR/vis/"
echo "💾 模型检查点: $OUTPUT_DIR/pruned/"
echo ""
echo "📈 训练统计:"
echo "   - 总迭代次数: 100,000"
echo "   - 等效epoch数: ~256"
echo "   - Batch size: 128"
echo "   - 学习率: 2e-4"
