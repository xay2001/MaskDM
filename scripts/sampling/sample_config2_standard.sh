#!/bin/bash
# Config2 Standard Model Sampling Script
# Based on sample_ddpm_cifar10_magnitude_pruned.sh

python ddpm_sample.py \
    --output_dir run/sample/maskpro/config2_standard \
    --batch_size 128 \
    --model_path Maskpro/train_result/config2_standard/lr1.0_epoch2000_logits5.0_size20000_diffusion/checkpoint \
    --total_samples 50000 \
    --ddim_steps 100 \
    --skip_type uniform \
    --seed 42
