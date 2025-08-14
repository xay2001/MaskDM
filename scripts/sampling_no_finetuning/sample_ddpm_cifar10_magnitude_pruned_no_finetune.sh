python ddpm_sample.py \
--output_dir run/sample/magnitude/ddpm_cifar10_pruned_no_finetune \
--batch_size 128 \
--pruned_model_ckpt run/pruned/magnitude/ddpm_cifar10_pruned/pruned/unet_pruned.pth \
--model_path run/pruned/magnitude/ddpm_cifar10_pruned \
--skip_type uniform
