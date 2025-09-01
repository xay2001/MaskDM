#!/bin/bash
# Training Configuration 3: Aggressive learning, full dataset
# 积极学习率，完整数据集，追求最佳效果
CUDA_VISIBLE_DEVICES=0 python train_diffusion.py \
      --original_model "../pretrained/ddpm_ema_cifar10" \
      --dataset "cifar10" \
      --lr 2.0 \
      --epoch 3000 \
      --logits 7.0 \
      --dataset_size 50000 \
      --batch_size 32 \
      --max_step 25000 \
      --targets all \
      --save \
      --output_dir "results_diffusion_config3_aggressive"
