#!/bin/bash
# Test Configuration 4: Efficient evaluation with large batch
# 高效评估，大批次设置对应的测试
python test_ddpm_maskpro.py \
    --checkpoint_path "results_diffusion_config4_efficient" \
    --original_model ../pretrained/ddpm_ema_cifar10 \
    --num_inference_steps 150 \
    --num_samples 1500 \
    --batch_size 50 \
    --compute_fid \
    --output_dir test_results_config4_efficient
