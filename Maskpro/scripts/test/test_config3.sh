#!/bin/bash
# Test Configuration 3: Comprehensive evaluation with aggressive settings
# 全面评估，积极设置对应的测试
python test_ddpm_maskpro.py \
    --checkpoint_path "train_result/config3_aggressive/checkpoint" \
    --original_model ../pretrained/ddpm_ema_cifar10 \
    --num_inference_steps 200 \
    --num_samples 2000 \
    --batch_size 20 \
    --compute_fid \
    --output_dir test_results_config3_aggressive
