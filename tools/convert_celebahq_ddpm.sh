#!/bin/bash
# Download google/ddpm-celebahq-256 model from Hugging Face
# This model is already in diffusers format, so no conversion needed
# Using HF-Mirror for faster download in China

mkdir -p pretrained

# 设置 HF-Mirror 镜像站环境变量
export HF_ENDPOINT=https://hf-mirror.com

python - <<EOF
from diffusers import DDPMPipeline
import os

print("Downloading google/ddpm-celebahq-256 model from HF-Mirror...")
model_id = "google/ddpm-celebahq-256"

# Download and save the model
pipeline = DDPMPipeline.from_pretrained(model_id)
save_path = "pretrained/ddpm_ema_celebahq_256"

print(f"Saving model to {save_path}...")
pipeline.save_pretrained(save_path)

print("Model downloaded successfully!")
print(f"Model saved at: {save_path}")
EOF

