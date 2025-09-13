#!/bin/bash
# Test Configuration 2: Standard evaluation with medium settings
# 标准评估，中等设置对应的测试
python test_ddpm_maskpro.py \
    --checkpoint_path "train_result/config2_standard/lr1.0_epoch2000_logits5.0_size20000_diffusion/checkpoint" \
    --original_model ../pretrained/ddpm_ema_cifar10 \
    --num_inference_steps 150 \
    --num_samples 1000 \
    --batch_size 32 \
    --compute_fid \
    --output_dir test_results_config2_standard
