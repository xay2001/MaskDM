#!/bin/bash
# Test Configuration 3: Comprehensive evaluation with aggressive settings
# 全面评估，积极设置对应的测试
python test_ddpm_maskpro.py \
    --checkpoint_path "results_diffusion_config3_aggressive" \
    --original_model ../pretrained/ddpm_ema_cifar10 \
    --num_inference_steps 200 \
    --num_samples 2000 \
    --batch_size 20 \
    --compute_fid \
    --output_dir test_results_config3_aggressive
