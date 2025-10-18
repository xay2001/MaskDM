#!/bin/bash
# 手动下载 google/ddpm-celebahq-256 模型的所有文件
# 使用 wget 从 HF-Mirror 直接下载

set -e  # 遇到错误立即退出

echo "📥 开始下载 google/ddpm-celebahq-256 模型..."
echo "使用 HF-Mirror 镜像站: https://hf-mirror.com"

# 创建目录结构
MODEL_DIR="pretrained/ddpm_ema_celebahq_256"
mkdir -p "$MODEL_DIR/unet"
mkdir -p "$MODEL_DIR/scheduler"

BASE_URL="https://hf-mirror.com/google/ddpm-celebahq-256/resolve/main"

echo ""
echo "1️⃣ 下载根目录配置文件..."
wget -nc -O "$MODEL_DIR/model_index.json" "$BASE_URL/model_index.json"

echo ""
echo "2️⃣ 下载 UNet 模型文件..."
wget -nc -O "$MODEL_DIR/unet/config.json" "$BASE_URL/unet/config.json"
wget -nc -O "$MODEL_DIR/unet/diffusion_pytorch_model.bin" "$BASE_URL/unet/diffusion_pytorch_model.bin"

echo ""
echo "3️⃣ 下载 Scheduler 配置文件..."
wget -nc -O "$MODEL_DIR/scheduler/scheduler_config.json" "$BASE_URL/scheduler/scheduler_config.json"

echo ""
echo "✅ 所有文件下载完成！"
echo "📁 模型保存位置: $MODEL_DIR"
echo ""
echo "📂 目录结构："
tree "$MODEL_DIR" 2>/dev/null || ls -R "$MODEL_DIR"

