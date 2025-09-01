#!/bin/bash
# Training Configuration 2: Standard learning, medium dataset
# 标准学习率，中等数据集，平衡效果
CUDA_VISIBLE_DEVICES=0 python train_diffusion.py \
      --original_model "../pretrained/ddpm_ema_cifar10" \
      --dataset "cifar10" \
      --lr 1.0 \
      --epoch 2000 \
      --logits 5.0 \
      --dataset_size 20000 \
      --batch_size 64 \
      --max_step 15000 \
      --targets all \
      --save \
      --output_dir "results_diffusion_config2_standard"
