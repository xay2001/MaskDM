#!/bin/bash

# 生成样本
python ddpm_sample.py \
--output_dir fid_samples/1019_cifar10_bs64_lr1_logit5 \
--batch_size 128 \
--total_samples 50000 \
--model_path /data/xay/MaskDM/finetuned_results/1019_cifar10_bs64_lr1_logit5 \
--ddim_steps 100 \
--skip_type uniform

# 计算FID
python fid_score.py \
fid_samples/1019_cifar10_bs64_lr1_logit5 \
run/fid_stats_cifar10.npz \
--batch-size 50 \
--num_samples 50000 \
--res 32 \
--dataset_name cifar10 \
--device cuda

