#!/bin/bash
# Training Configuration 5: Fine-tuning focused, high logits
# 精细调优，高logits倍数，适合细节优化
CUDA_VISIBLE_DEVICES=0 python train_diffusion.py \
      --original_model "../pretrained/ddpm_ema_cifar10" \
      --dataset "cifar10" \
      --lr 0.8 \
      --epoch 4000 \
      --logits 10.0 \
      --dataset_size 30000 \
      --batch_size 32 \
      --max_step 30000 \
      --targets all \
      --save \
      --output_dir "results_diffusion_config5_finetuning"
