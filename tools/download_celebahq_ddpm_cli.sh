#!/bin/bash
# 使用 huggingface-cli 下载 google/ddpm-celebahq-256 模型
# 推荐使用此方法，更稳定可靠

# 设置 HF-Mirror 镜像站
export HF_ENDPOINT=https://hf-mirror.com

echo "使用 HF-Mirror 镜像站下载模型..."
echo "模型: google/ddpm-celebahq-256"
echo "保存路径: pretrained/ddpm_ema_celebahq_256"

# 创建目录
mkdir -p pretrained

# 使用 huggingface-cli 下载（推荐方法）
huggingface-cli download \
    --resume-download \
    google/ddpm-celebahq-256 \
    --local-dir pretrained/ddpm_ema_celebahq_256 \
    --local-dir-use-symlinks False

if [ $? -eq 0 ]; then
    echo "✅ 模型下载成功！"
    echo "📁 保存位置: pretrained/ddpm_ema_celebahq_256"
else
    echo "❌ 下载失败，请检查网络连接"
    exit 1
fi

