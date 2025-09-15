#!/bin/bash

# 使用ddpm_train.py直接微调剪枝模型脚本

# 设置路径
PRUNED_MODEL_PATH="/data/xay/MaskDM/Maskpro/train_result/config2_standard/lr1.0_epoch2000_logits5.0_size20000_diffusion/checkpoint"
OUTPUT_DIR="/data/xay/MaskDM/finetuned_results/ddpm_train_finetuned"

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "开始使用ddpm_train.py微调剪枝模型..."
echo "剪枝模型路径: $PRUNED_MODEL_PATH"
echo "输出目录: $OUTPUT_DIR"

# 使用ddpm_train.py进行微调
# 关键参数：
# --model_path: 剪枝模型路径
# --num_iters: 微调步数（相比从头训练要少）
# --learning_rate: 较小的学习率
# --use_ema: 使用EMA
python ddpm_train.py \
    --model_path $PRUNED_MODEL_PATH \
    --dataset cifar10 \
    --output_dir $OUTPUT_DIR \
    --resolution 32 \
    --train_batch_size 32 \
    --eval_batch_size 16 \
    --num_iters 2000 \
    --learning_rate 1e-5 \
    --lr_scheduler cosine \
    --lr_warmup_steps 100 \
    --use_ema \
    --save_model_steps 500 \
    --gradient_accumulation_steps 2 \
    --adam_weight_decay 1e-6 \
    --mixed_precision fp16 \
    --logger wandb \
    --overwrite_output_dir

echo "微调完成！"
echo "输出目录: $OUTPUT_DIR"
echo "可以查看 $OUTPUT_DIR/vis/ 中的生成样本"
