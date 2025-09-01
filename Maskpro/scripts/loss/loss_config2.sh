#!/bin/bash
# Configuration 2: Medium dataset, larger batch size
# 平衡实验速度和数据覆盖率
python inference_loss_diffusion.py \
      --original_model "../pretrained/ddpm_ema_cifar10" \
      --dataset "cifar10" \
      --dataset_size 20000 \
      --batch_size 64 \
      --max_batches 313 \
      --device "cuda:0" \
      --targets all
