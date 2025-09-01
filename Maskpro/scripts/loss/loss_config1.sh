#!/bin/bash
# Configuration 1: Small dataset, standard batch size
# 适合快速实验和调试
python inference_loss_diffusion.py \
      --original_model "../pretrained/ddpm_ema_cifar10" \
      --dataset "cifar10" \
      --dataset_size 5000 \
      --batch_size 16 \
      --max_batches 313 \
      --device "cuda:0" \
      --targets all
