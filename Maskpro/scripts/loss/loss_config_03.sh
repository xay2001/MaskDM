#!/bin/bash
# Baseline loss computation for 0.3 pruning rate
# 计算0.3剪枝率的基线损失

python inference_loss_diffusion.py \
    --original_model "../pretrained/ddpm_ema_cifar10" \
    --dataset cifar10 \
    --dataset_size 20000 \
    --batch_size 64 \
    --device cuda:0 \
    --max_batches 313 \
    --initial_mask_path "initial_mask_diffusion_30" \
    --learned_mask_path "learned_mask_diffusion_30" \
    --targets all
