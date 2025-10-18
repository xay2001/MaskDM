python ddpm_weight_prune.py \
    --dataset celebahq \
    --model_path pretrained/ddpm_ema_celebahq_256 \
    --save_path run/pruned/weight_magnitude/ddpm_celebahq_weight_pruned \
    --pruning_ratio $1 \
    --batch_size 128 \
    --pruner magnitude \
    --device cuda:0 \
