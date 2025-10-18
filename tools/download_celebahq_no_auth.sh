#!/bin/bash
# 下载 google/ddpm-celebahq-256 模型（移除认证信息）
# 解决 401 Unauthorized 错误

# 设置 HF-Mirror 镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 临时禁用 HF Token（避免 401 错误）
unset HF_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

mkdir -p pretrained

echo "📥 使用 HF-Mirror 下载模型（无认证模式）..."
echo "模型: google/ddpm-celebahq-256"
echo "保存路径: pretrained/ddpm_ema_celebahq_256"

python - <<'EOF'
from diffusers import DDPMPipeline
import os

# 禁用隐式 token
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'

print("\n下载中...")
model_id = "google/ddpm-celebahq-256"

try:
    # 使用 use_auth_token=False 明确禁用认证
    pipeline = DDPMPipeline.from_pretrained(
        model_id,
        use_auth_token=False
    )
    save_path = "pretrained/ddpm_ema_celebahq_256"
    
    print(f"保存模型到 {save_path}...")
    pipeline.save_pretrained(save_path)
    
    print("\n✅ 模型下载成功！")
    print(f"📁 保存位置: {save_path}")
except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    exit(1)
EOF

