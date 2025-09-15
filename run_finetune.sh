#!/bin/bash

# 剪枝模型微调启动脚本

# 激活conda环境
source /home/xay/sharedspace/conda/etc/profile.d/conda.sh
conda activate prunedm

# 设置路径
PRUNED_MODEL_PATH="/data/xay/MaskDM/Maskpro/train_result/config2_standard/lr1.0_epoch2000_logits5.0_size20000_diffusion/checkpoint"
OUTPUT_DIR="/data/xay/MaskDM/finetuned_results/config2_standard_finetuned"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行微调
python /data/xay/MaskDM/finetune_pruned_model.py \
    --pruned_model_path $PRUNED_MODEL_PATH \
    --dataset cifar10 \
    --output_dir $OUTPUT_DIR \
    --resolution 32 \
    --train_batch_size 32 \
    --eval_batch_size 16 \
    --num_iters 3000 \
    --learning_rate 5e-6 \
    --lr_scheduler cosine \
    --lr_warmup_steps 150 \
    --use_ema \
    --save_model_steps 500 \
    --eval_steps 250 \
    --logger tensorboard \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 2 \
    --adam_weight_decay 1e-6 \
    --use_swanlab \
    --overwrite_output_dir

echo "微调启动完成！"
echo "输出目录: $OUTPUT_DIR"
echo "可以使用 tensorboard --logdir $OUTPUT_DIR/logs 查看训练进度"
