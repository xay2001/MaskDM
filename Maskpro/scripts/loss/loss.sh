python inference_loss_diffusion.py \
      --original_model "../pretrained/ddpm_ema_cifar10" \
      --dataset "cifar10" \
      --dataset_size 50000 \
      --batch_size 32 \
      --max_batches 1562 \
      --device "cuda:0" \
      --targets all