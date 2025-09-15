#!/bin/bash
# Training Configuration for 0.3 pruning rate
# 使用0.3剪枝率mask的训练配置

CUDA_VISIBLE_DEVICES=0 python train_diffusion.py \
    --original_model "../pretrained/ddpm_ema_cifar10" \
    --dataset "cifar10" \
    --lr 1.0 \
    --epoch 1000 \
    --logits 5.0 \
    --dataset_size 20000 \
    --batch_size 64 \
    --max_step 15000 \
    --targets all \
    --save \
    --initial_mask_path "initial_mask_diffusion_30" \
    --learned_mask_path "learned_mask_diffusion_30" \
    --output_dir "results_diffusion_30" \
    --project_name "DDPM-MaskPro-30" \
    --experiment_name "ddpm_maskpro_30_pruning"
