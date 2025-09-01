#!/bin/bash
# Training Configuration 4: High efficiency, large batch
# 高效训练，大批次，GPU优化
CUDA_VISIBLE_DEVICES=0 python train_diffusion.py \
      --original_model "../pretrained/ddpm_ema_cifar10" \
      --dataset "cifar10" \
      --lr 1.5 \
      --epoch 1500 \
      --logits 6.0 \
      --dataset_size 25000 \
      --batch_size 128 \
      --max_step 12000 \
      --targets all \
      --save \
      --output_dir "results_diffusion_config4_efficient"
