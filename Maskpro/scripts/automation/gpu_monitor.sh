#!/bin/bash
# GPU监控脚本: 实时监控实验进度
# GPU Monitor: Real-time experiment progress monitoring

LOG_DIR="/data/xay/MaskDM/Maskpro/logs"

echo "=========================================="
echo "MaskDM 实验监控面板"
echo "时间: $(date)"
echo "=========================================="

# 检查NVIDIA工具
if ! command -v nvidia-smi &> /dev/null; then
    echo "警告: nvidia-smi 未找到，无法显示GPU状态"
fi

# 监控函数
monitor_experiments() {
    while true; do
        clear
        echo "=========================================="
        echo "MaskDM 实验监控面板 - $(date)"
        echo "=========================================="
        
        # GPU状态
        if command -v nvidia-smi &> /dev/null; then
            echo ""
            echo "GPU状态:"
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while IFS=, read gpu_id name util mem_used mem_total temp; do
                mem_percent=$(( mem_used * 100 / mem_total ))
                echo "  GPU ${gpu_id}: ${util}% 利用率, ${mem_percent}% 内存 (${mem_used}MB/${mem_total}MB), ${temp}°C"
            done
        fi
        
        echo ""
        echo "实验状态:"
        
        # 检查配置状态
        for config in 1 2 3 4 5; do
            log_file="${LOG_DIR}/config${config}.log"
            if [ -f "$log_file" ]; then
                # 获取最后几行来判断状态
                last_lines=$(tail -n 5 "$log_file" 2>/dev/null)
                
                if echo "$last_lines" | grep -q "自动化流程完成"; then
                    status="✓ 完成"
                elif echo "$last_lines" | grep -q "失败\|错误\|error"; then
                    status="✗ 失败"
                elif echo "$last_lines" | grep -q "步骤[123]"; then
                    current_step=$(echo "$last_lines" | grep "步骤[123]" | tail -n 1 | sed 's/.*步骤\([123]\).*/\1/')
                    case $current_step in
                        1) status="🔄 生成损失中" ;;
                        2) status="🔄 训练中" ;;
                        3) status="🔄 测试中" ;;
                        *) status="🔄 运行中" ;;
                    esac
                else
                    status="🔄 运行中"
                fi
                
                # 检查进程是否还在运行
                if ! pgrep -f "auto_config${config}.sh" > /dev/null; then
                    if [ "$status" != "✓ 完成" ]; then
                        status="⏹️ 已停止"
                    fi
                fi
                
                echo "  Config ${config}: ${status}"
            else
                echo "  Config ${config}: ⬜ 未启动"
            fi
        done
        
        echo ""
        echo "结果目录:"
        
        # 检查结果目录
        train_results=$(find /data/xay/MaskDM/Maskpro/train_result -maxdepth 1 -type d 2>/dev/null | wc -l)
        test_results=$(find /data/xay/MaskDM/Maskpro/test_result -maxdepth 1 -type d 2>/dev/null | wc -l)
        
        # 减去1是因为find会包含父目录
        train_results=$((train_results - 1))
        test_results=$((test_results - 1))
        
        echo "  训练结果: ${train_results} 个"
        echo "  测试结果: ${test_results} 个"
        
        echo ""
        echo "操作选项:"
        echo "  q: 退出监控"
        echo "  l: 查看日志"
        echo "  k: 终止实验"
        echo "  r: 刷新 (自动每10秒刷新)"
        echo ""
        
        # 非阻塞读取用户输入
        read -t 10 -n 1 input
        case $input in
            q|Q)
                echo "退出监控..."
                break
                ;;
            l|L)
                echo ""
                echo "选择要查看的配置日志 (1-5):"
                read -n 1 log_choice
                if [ "$log_choice" -ge 1 ] && [ "$log_choice" -le 5 ]; then
                    if [ -f "${LOG_DIR}/config${log_choice}.log" ]; then
                        echo ""
                        echo "显示 Config ${log_choice} 日志 (按 q 退出):"
                        less "${LOG_DIR}/config${log_choice}.log"
                    else
                        echo "日志文件不存在: ${LOG_DIR}/config${log_choice}.log"
                        sleep 2
                    fi
                fi
                ;;
            k|K)
                echo ""
                echo "终止所有实验进程..."
                pkill -f "auto_config"
                echo "已发送终止信号"
                sleep 2
                ;;
        esac
    done
}

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 启动监控
monitor_experiments
