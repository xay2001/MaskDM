#!/bin/bash

# LDM Weight Pruning Script
# This script performs unstructured weight pruning on LDM's UNet
# VQVAE is fully preserved to maintain reconstruction quality

python ldm_weight_prune.py \
--model_path /data/xay/MaskDM/pretrained/ldm_celebahq_256 \
--save_path run/pruned/ldm_celeba_weight_pruned_0.5 \
--pruning_ratio 0.5 \
--pruner magnitude \
--device cuda:0 \
--batch_size 4

