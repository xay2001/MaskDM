#!/bin/bash

# ====================================
# 批量运行 CelebA-HQ MaskPro 训练实验
# 全部并行运行，使用 GPU 1、2、3
# ====================================

# 设置工作目录
cd /data/xay/MaskDM/Maskpro

# 创建日志目录
LOG_DIR="logs/batch_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "启动批量训练实验（全部并行）"
echo "日志目录: $LOG_DIR"
echo "=========================================="

# ==========================================
# GPU 1: 实验3（完整训练，最重要）
# ==========================================
{
    echo "[GPU 1] 开始实验3 - 完整训练（预计1-2天）"
    CUDA_VISIBLE_DEVICES=1 python train_diffusion.py \
        --original_model "../pretrained/ddpm_ema_celebahq_256" \
        --dataset "../data/CelebAMask-HQ/CelebA-HQ-img" \
        --lr 1.0 \
        --epoch 2000 \
        --logits 5.0 \
        --dataset_size 5000 \
        --batch_size 32 \
        --max_step 320000 \
        --targets all \
        --save \
        --output_dir "train_result/celebahq_magnitude_full_lr1.0" \
        --initial_mask_path "initial_mask_diffusion/celebahq_magnitude" \
        --learned_mask_path "learned_mask_diffusion/celebahq_magnitude" \
        --device "cuda:0" \
        --experiment_name "celebahq_magnitude_full_lr1.0_epoch2000" \
        2>&1 | tee "$LOG_DIR/exp3_full_training.log"
    echo "[GPU 1] 实验3完成"
} &

# ==========================================
# GPU 1: 实验7（中速训练）
# ==========================================
{
    echo "[GPU 1] 开始实验7 - 中速训练（预计4-6小时）"
    CUDA_VISIBLE_DEVICES=1 python train_diffusion.py \
        --original_model "../pretrained/ddpm_ema_celebahq_256" \
        --dataset "../data/CelebAMask-HQ/CelebA-HQ-img" \
        --lr 1.5 \
        --epoch 1200 \
        --logits 5.0 \
        --dataset_size 3000 \
        --batch_size 32 \
        --max_step 120000 \
        --targets all \
        --save \
        --output_dir "train_result/celebahq_magnitude_medium_lr1.5" \
        --initial_mask_path "initial_mask_diffusion/celebahq_magnitude" \
        --learned_mask_path "learned_mask_diffusion/celebahq_magnitude" \
        --device "cuda:0" \
        --experiment_name "celebahq_magnitude_medium_lr1.5" \
        2>&1 | tee "$LOG_DIR/exp7_medium_training.log"
    echo "[GPU 1] 实验7完成"
} &

# ==========================================
# GPU 2: 实验4（快速高效训练）
# ==========================================
{
    echo "[GPU 2] 开始实验4 - 快速高效训练（预计12-18小时）"
    CUDA_VISIBLE_DEVICES=2 python train_diffusion.py \
        --original_model "../pretrained/ddpm_ema_celebahq_256" \
        --dataset "../data/CelebAMask-HQ/CelebA-HQ-img" \
        --lr 2.0 \
        --epoch 1000 \
        --logits 5.0 \
        --dataset_size 8000 \
        --batch_size 32 \
        --max_step 100000 \
        --targets all \
        --save \
        --output_dir "train_result/celebahq_magnitude_fast_lr2.0" \
        --initial_mask_path "initial_mask_diffusion/celebahq_magnitude" \
        --learned_mask_path "learned_mask_diffusion/celebahq_magnitude" \
        --device "cuda:0" \
        --experiment_name "celebahq_magnitude_fast_lr2.0_epoch1000" \
        2>&1 | tee "$LOG_DIR/exp4_fast_training.log"
    echo "[GPU 2] 实验4完成"
} &

# ==========================================
# GPU 2: 实验6（快速标准版）
# ==========================================
{
    echo "[GPU 2] 开始实验6 - 快速标准版（预计3-5小时）"
    CUDA_VISIBLE_DEVICES=2 python train_diffusion.py \
        --original_model "../pretrained/ddpm_ema_celebahq_256" \
        --dataset "../data/CelebAMask-HQ/CelebA-HQ-img" \
        --lr 2.0 \
        --epoch 800 \
        --logits 5.0 \
        --dataset_size 3200 \
        --batch_size 64 \
        --max_step 50000 \
        --targets all \
        --save \
        --output_dir "train_result/celebahq_magnitude_fast_lr2.0_bs64" \
        --initial_mask_path "initial_mask_diffusion/celebahq_magnitude" \
        --learned_mask_path "learned_mask_diffusion/celebahq_magnitude" \
        --device "cuda:0" \
        --experiment_name "celebahq_magnitude_fast_lr2.0_bs64" \
        2>&1 | tee "$LOG_DIR/exp6_fast_standard.log"
    echo "[GPU 2] 实验6完成"
} &

# ==========================================
# GPU 3: 实验5（超快速验证）
# ==========================================
{
    echo "[GPU 3] 开始实验5 - 超快速验证（预计1-2小时）"
    CUDA_VISIBLE_DEVICES=3 python train_diffusion.py \
        --original_model "../pretrained/ddpm_ema_celebahq_256" \
        --dataset "../data/CelebAMask-HQ/CelebA-HQ-img" \
        --lr 3.0 \
        --epoch 500 \
        --logits 5.0 \
        --dataset_size 2000 \
        --batch_size 64 \
        --max_step 20000 \
        --targets all \
        --save \
        --output_dir "train_result/celebahq_magnitude_ultrafast_lr3.0" \
        --initial_mask_path "initial_mask_diffusion/celebahq_magnitude" \
        --learned_mask_path "learned_mask_diffusion/celebahq_magnitude" \
        --device "cuda:0" \
        --experiment_name "celebahq_magnitude_ultrafast_lr3.0" \
        2>&1 | tee "$LOG_DIR/exp5_ultrafast.log"
    echo "[GPU 3] 实验5完成"
} &

# 等待所有后台任务完成
wait

echo "=========================================="
echo "所有实验已完成！"
echo "日志位置: $LOG_DIR"
echo "结果目录: train_result/"
echo "=========================================="

# 生成实验总结
cat > "$LOG_DIR/experiment_summary.txt" << EOF
批量训练实验总结
执行时间: $(date)

实验配置:
-----------
实验3 (GPU 1): 完整训练 - lr=1.0, epochs=2000, dataset=5000, steps=320K
实验7 (GPU 1): 中速训练 - lr=1.5, epochs=1200, dataset=3000, steps=120K
实验4 (GPU 2): 快速高效 - lr=2.0, epochs=1000, dataset=8000, steps=100K
实验6 (GPU 2): 快速标准 - lr=2.0, epochs=800, dataset=3200, bs=64, steps=50K
实验5 (GPU 3): 超快验证 - lr=3.0, epochs=500, dataset=2000, bs=64, steps=20K

输出目录:
-----------
实验3: train_result/celebahq_magnitude_full_lr1.0
实验4: train_result/celebahq_magnitude_fast_lr2.0
实验5: train_result/celebahq_magnitude_ultrafast_lr3.0
实验6: train_result/celebahq_magnitude_fast_lr2.0_bs64
实验7: train_result/celebahq_magnitude_medium_lr1.5

日志文件:
-----------
$(ls -lh "$LOG_DIR"/*.log)
EOF

echo "实验总结已保存到: $LOG_DIR/experiment_summary.txt"

