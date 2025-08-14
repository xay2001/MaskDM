python ddpm_weight_prune.py \
--dataset cifar10 \
--model_path pretrained/ddpm_ema_cifar10 \
--save_path run/pruned/weight_random/ddpm_cifar10_weight_pruned \
--pruning_ratio $1 \
--batch_size 128 \
--pruner random \
--device cuda:2 \