#!/bin/bash
# 为 MaskDM 项目安装依赖包 (适配 Python 3.11)

set -e

echo "=========================================="
echo "MaskDM 项目依赖安装脚本"
echo "Python 版本: $(python --version)"
echo "Conda 环境: $CONDA_DEFAULT_ENV"
echo "=========================================="

# 配置清华源
echo "配置pip清华镜像源..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# # Step 1: 安装核心深度学习框架 (Python 3.11兼容版本)
# echo "Step 1: 安装 PyTorch 和相关包..."
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# # Step 2: 安装 diffusers 和 HuggingFace 相关
# echo "Step 2: 安装 Diffusers 和 HuggingFace..."
# pip install diffusers transformers  accelerate datasets

# Step 3: 安装科学计算包 (兼容Python 3.11版本)
echo "Step 3: 安装科学计算包..."
pip install "numpy>=1.24.0" "scipy>=1.9.0" "matplotlib>=3.5.0" "seaborn>=0.11.0"

# Step 4: 安装图像处理和评估
echo "Step 4: 安装图像处理和评估工具..."
pip install "pillow>=9.0.0" opencv-python-headless

# Step 5: 安装剪枝和工具包
echo "Step 5: 安装剪枝和工具包..."
pip install "torch-pruning>=1.3.0"

# Step 6: 安装实验监控 (可选)
echo "Step 6: 安装实验监控..."
pip install wandb swanlab tqdm

# Step 7: 安装其他必要工具
echo "Step 7: 安装其他工具..."
pip install pyyaml requests safetensors protobuf psutil regex

# Step 8: 安装项目特定包
echo "Step 8: 安装项目特定包..."
pip install "accelerate>=0.20.0" datasets "transformers>=4.30.0"

# Step 9: 安装FID计算相关包
echo "Step 9: 安装FID计算工具..."
pip install "scipy>=1.9.0"

echo "=========================================="
echo "安装完成！"
echo "=========================================="

echo "验证关键包安装:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"

echo ""
echo "检查GPU可用性:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
