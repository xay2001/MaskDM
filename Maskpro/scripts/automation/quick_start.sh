#!/bin/bash
# 快速启动脚本: 一键开始实验
# Quick Start: One-click experiment launcher

set -e

BASE_DIR="/data/xay/MaskDM/Maskpro"
SCRIPT_DIR="${BASE_DIR}/scripts/automation"

echo "=========================================="
echo "MaskDM 快速启动器"
echo "=========================================="

# 切换到正确目录
cd "${BASE_DIR}"

# 检查GPU可用性
echo "检查GPU状态..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo ""
else
    echo "警告: 无法检查GPU状态"
fi

# 检查预训练模型
echo "检查预训练模型..."
if [ ! -d "../pretrained/ddpm_ema_cifar10" ]; then
    echo "❌ 错误: 预训练模型未找到"
    echo "请确保 ../pretrained/ddpm_ema_cifar10 目录存在"
    exit 1
else
    echo "✅ 预训练模型检查通过"
fi

# 检查Python脚本
echo "检查必要文件..."
required_files=(
    "inference_loss_diffusion.py"
    "train_diffusion.py" 
    "test_ddpm_maskpro.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 错误: 缺少文件 $file"
        exit 1
    fi
done
echo "✅ 必要文件检查通过"

# 创建结果目录
echo "创建结果目录..."
mkdir -p train_result test_result logs
echo "✅ 结果目录创建完成"

# 设置脚本权限
echo "设置脚本权限..."
chmod +x "${SCRIPT_DIR}"/*.sh
echo "✅ 权限设置完成"

echo ""
echo "=========================================="
echo "选择启动模式:"
echo "=========================================="
echo "1. 🚀 快速验证 (Config 1, ~30分钟)"
echo "2. ⚖️  标准实验 (Config 1+2, ~2小时)" 
echo "3. 🔥 完整实验 (所有配置, ~8-12小时)"
echo "4. 🎯 自定义选择"
echo "5. 📊 仅启动监控面板"
echo ""
read -p "请选择 (1-5): " mode

case $mode in
    1)
        echo ""
        echo "🚀 启动快速验证模式..."
        echo "   配置: Config 1 (快速验证)"
        echo "   GPU: cuda:2"
        echo "   预计时间: ~30分钟"
        echo ""
        
        # 启动配置1和监控
        nohup bash "${SCRIPT_DIR}/auto_config1.sh" > logs/config1.log 2>&1 &
        CONFIG_PID=$!
        echo "✅ Config 1 已启动 (PID: $CONFIG_PID)"
        
        echo ""
        echo "监控命令:"
        echo "  tail -f logs/config1.log"
        echo "  bash scripts/automation/gpu_monitor.sh"
        ;;
        
    2)
        echo ""
        echo "⚖️ 启动标准实验模式..."
        echo "   配置: Config 1 (快速验证) + Config 2 (标准平衡)"
        echo "   GPU: cuda:2 + cuda:3"
        echo "   预计时间: ~2小时"
        echo ""
        
        # 并行启动配置1和2
        nohup bash "${SCRIPT_DIR}/auto_config1.sh" > logs/config1.log 2>&1 &
        CONFIG1_PID=$!
        nohup bash "${SCRIPT_DIR}/auto_config2.sh" > logs/config2.log 2>&1 &
        CONFIG2_PID=$!
        
        echo "✅ Config 1 已启动 (PID: $CONFIG1_PID)"
        echo "✅ Config 2 已启动 (PID: $CONFIG2_PID)"
        
        echo ""
        echo "监控命令:"
        echo "  bash scripts/automation/gpu_monitor.sh"
        ;;
        
    3)
        echo ""
        echo "🔥 启动完整实验模式..."
        echo "   配置: 所有5个配置"
        echo "   GPU: cuda:2 + cuda:3 (并行)"
        echo "   预计时间: ~8-12小时"
        echo ""
        read -p "确认启动完整实验？(y/N): " confirm
        
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            bash "${SCRIPT_DIR}/run_all_experiments.sh"
        else
            echo "已取消"
            exit 0
        fi
        ;;
        
    4)
        echo ""
        echo "🎯 自定义选择模式..."
        bash "${SCRIPT_DIR}/run_all_experiments.sh"
        ;;
        
    5)
        echo ""
        echo "📊 启动监控面板..."
        bash "${SCRIPT_DIR}/gpu_monitor.sh"
        exit 0
        ;;
        
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "实验已启动！"
echo "=========================================="
echo ""
echo "📁 结果位置:"
echo "   训练: ${BASE_DIR}/train_result/"
echo "   测试: ${BASE_DIR}/test_result/" 
echo "   日志: ${BASE_DIR}/logs/"
echo ""
echo "🔍 监控命令:"
echo "   bash scripts/automation/gpu_monitor.sh"
echo ""
echo "⏹️  停止命令:"
echo "   pkill -f auto_config"
echo ""
echo "实验启动完成! 🎉"
