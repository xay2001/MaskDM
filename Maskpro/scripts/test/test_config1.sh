#!/bin/bash
# Test Configuration 1: Quick validation with conservative settings
# 快速验证，保守设置对应的测试
python test_ddpm_maskpro.py \
    --checkpoint_path "results_diffusion_config1_conservative" \
    --original_model ../pretrained/ddpm_ema_cifar10 \
    --num_inference_steps 100 \
    --num_samples 500 \
    --batch_size 16 \
    --compute_fid \
    --output_dir test_results_config1_conservative
