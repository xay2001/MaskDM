#!/bin/bash
# Training Configuration 1: Conservative learning, quick iteration
# 保守学习率，快速迭代，适合初期调试
CUDA_VISIBLE_DEVICES=0 python train_diffusion.py \
      --original_model "../pretrained/ddpm_ema_cifar10" \
      --dataset "cifar10" \
      --lr 0.5 \
      --epoch 1000 \
      --logits 3.0 \
      --dataset_size 5000 \
      --batch_size 16 \
      --max_step 5000 \
      --targets all \
      --save \
      --output_dir "results_diffusion_config1_conservative"
