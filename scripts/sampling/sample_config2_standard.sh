#!/bin/bash
# Config2 Standard Model Sampling Script
# Based on sample_ddpm_cifar10_magnitude_pruned.sh

python ddpm_sample.py \
    --output_dir run/sample/maskpro/config2_standard \
    --batch_size 128 \
    --model_path /data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2 \
    --pruned_model_ckpt /data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2/pruned/unet_ema_pruned.pth \
    --total_samples 50000 \
    --ddim_steps 100 \
    --skip_type uniform \
    --seed 42
