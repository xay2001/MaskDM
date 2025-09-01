#!/bin/bash
# Test Configuration 5: High-quality evaluation for fine-tuned model
# 高质量评估，精细调优模型对应的测试
python test_ddpm_maskpro.py \
    --checkpoint_path "results_diffusion_config5_finetuning" \
    --original_model ../pretrained/ddpm_ema_cifar10 \
    --num_inference_steps 250 \
    --num_samples 2500 \
    --batch_size 25 \
    --compute_fid \
    --output_dir test_results_config5_finetuning
