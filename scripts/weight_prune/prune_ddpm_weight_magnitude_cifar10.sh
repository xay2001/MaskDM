python ddpm_weight_prune.py \
    --dataset cifar10 \
    --model_path pretrained/ddpm_ema_cifar10 \
    --save_path run/pruned/weight_magnitude/ddpm_cifar10_weight_pruned \
    --pruning_ratio $1 \
    --batch_size 128 \
    --pruner magnitude \
    --device cuda:0 \
