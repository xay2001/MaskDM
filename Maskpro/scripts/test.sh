python test_ddpm_maskpro.py \
    --checkpoint_path "results_diffusion_optimized_2/lr0.8_epoch2000_logits3.0_size8192_diffusion" \
    --original_model ../pretrained/ddpm_ema_cifar10 \
    --num_inference_steps 200 \
    --num_samples 1000 \
    --batch_size 10 \
    --compute_fid \
    --output_dir test_results_optimized_3