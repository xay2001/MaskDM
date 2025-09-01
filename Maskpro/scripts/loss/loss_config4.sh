#!/bin/bash
# Configuration 4: Large batch size for GPU efficiency
# 大批次实验，充分利用GPU内存
python inference_loss_diffusion.py \
      --original_model "../pretrained/ddpm_ema_cifar10" \
      --dataset "cifar10" \
      --dataset_size 25000 \
      --batch_size 128 \
      --max_batches 196 \
      --device "cuda:0" \
      --targets all
