#!/bin/bash
# 主控脚本: 运行所有自动化实验配置
# Master script: Run all automated experiment configurations

BASE_DIR="/data/xay/MaskDM/Maskpro"
SCRIPT_DIR="${BASE_DIR}/scripts/automation"
LOG_DIR="${BASE_DIR}/logs"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# GPU分配策略
# GPU 2: Config 1, 3, 5 (奇数配置)
# GPU 3: Config 2, 4 (偶数配置)

echo "=========================================="
echo "MaskDM 自动化实验启动器"
echo "时间: $(date)"
echo "=========================================="

echo "GPU分配策略:"
echo "  GPU 2: Config 1 (快速验证), Config 3 (最佳性能), Config 5 (精细调优)"
echo "  GPU 3: Config 2 (标准平衡), Config 4 (GPU优化)"
echo ""

# 创建结果目录
cd "${BASE_DIR}"
mkdir -p train_result test_result

echo "选择运行模式:"
echo "1. 并行运行所有配置 (推荐，如果GPU资源充足)"
echo "2. 串行运行所有配置 (安全，避免资源冲突)"
echo "3. 选择性运行特定配置"
echo "4. 仅运行快速验证 (Config 1)"
echo ""
read -p "请输入选择 (1-4): " choice

case $choice in
    1)
        echo "启动并行模式..."
        echo "警告: 确保有足够的GPU内存！"
        
        # GPU 2上的配置 (后台运行)
        echo "[$(date)] 启动 Config 1 (GPU 2)..."
        nohup bash "${SCRIPT_DIR}/auto_config1.sh" > "${LOG_DIR}/config1.log" 2>&1 &
        CONFIG1_PID=$!
        
        echo "[$(date)] 启动 Config 3 (GPU 2)..."
        nohup bash "${SCRIPT_DIR}/auto_config3.sh" > "${LOG_DIR}/config3.log" 2>&1 &
        CONFIG3_PID=$!
        
        echo "[$(date)] 启动 Config 5 (GPU 2)..."
        nohup bash "${SCRIPT_DIR}/auto_config5.sh" > "${LOG_DIR}/config5.log" 2>&1 &
        CONFIG5_PID=$!
        
        # GPU 3上的配置 (后台运行)
        echo "[$(date)] 启动 Config 2 (GPU 3)..."
        nohup bash "${SCRIPT_DIR}/auto_config2.sh" > "${LOG_DIR}/config2.log" 2>&1 &
        CONFIG2_PID=$!
        
        echo "[$(date)] 启动 Config 4 (GPU 3)..."
        nohup bash "${SCRIPT_DIR}/auto_config4.sh" > "${LOG_DIR}/config4.log" 2>&1 &
        CONFIG4_PID=$!
        
        echo ""
        echo "所有配置已启动，进程ID:"
        echo "  Config 1: $CONFIG1_PID"
        echo "  Config 2: $CONFIG2_PID"
        echo "  Config 3: $CONFIG3_PID"
        echo "  Config 4: $CONFIG4_PID"
        echo "  Config 5: $CONFIG5_PID"
        echo ""
        echo "监控日志:"
        echo "  tail -f ${LOG_DIR}/config1.log"
        echo "  tail -f ${LOG_DIR}/config2.log"
        echo "  # 等等..."
        
        # 等待所有进程完成
        echo "等待所有配置完成..."
        wait $CONFIG1_PID $CONFIG2_PID $CONFIG3_PID $CONFIG4_PID $CONFIG5_PID
        echo "[$(date)] 所有并行配置完成！"
        ;;
        
    2)
        echo "启动串行模式..."
        
        for config in 1 2 3 4 5; do
            echo ""
            echo "[$(date)] 开始运行 Config ${config}..."
            bash "${SCRIPT_DIR}/auto_config${config}.sh" 2>&1 | tee "${LOG_DIR}/config${config}.log"
            
            if [ $? -eq 0 ]; then
                echo "[$(date)] Config ${config} 完成 ✓"
            else
                echo "[$(date)] Config ${config} 失败 ✗"
                read -p "是否继续下一个配置？(y/n): " continue_choice
                if [ "$continue_choice" != "y" ]; then
                    echo "实验中止。"
                    exit 1
                fi
            fi
        done
        echo "[$(date)] 所有串行配置完成！"
        ;;
        
    3)
        echo "选择要运行的配置:"
        echo "输入配置编号，用空格分隔 (例如: 1 3 5)"
        read -p "配置选择: " selected_configs
        
        for config in $selected_configs; do
            if [ -f "${SCRIPT_DIR}/auto_config${config}.sh" ]; then
                echo ""
                echo "[$(date)] 开始运行 Config ${config}..."
                bash "${SCRIPT_DIR}/auto_config${config}.sh" 2>&1 | tee "${LOG_DIR}/config${config}.log"
                
                if [ $? -eq 0 ]; then
                    echo "[$(date)] Config ${config} 完成 ✓"
                else
                    echo "[$(date)] Config ${config} 失败 ✗"
                fi
            else
                echo "配置 ${config} 不存在！"
            fi
        done
        ;;
        
    4)
        echo "运行快速验证配置..."
        bash "${SCRIPT_DIR}/auto_config1.sh" 2>&1 | tee "${LOG_DIR}/config1.log"
        ;;
        
    *)
        echo "无效选择！"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "实验完成总结"
echo "时间: $(date)"
echo "=========================================="

# 检查结果
echo "训练结果目录: ${BASE_DIR}/train_result/"
echo "测试结果目录: ${BASE_DIR}/test_result/"
echo "日志目录: ${LOG_DIR}/"
echo ""

echo "可用结果:"
for result_dir in "${BASE_DIR}/train_result"/*; do
    if [ -d "$result_dir" ]; then
        config_name=$(basename "$result_dir")
        echo "  训练: train_result/${config_name}"
        if [ -d "${BASE_DIR}/test_result/${config_name}" ]; then
            echo "  测试: test_result/${config_name}"
        fi
    fi
done

echo ""
echo "查看结果命令示例:"
echo "  ls -la train_result/"
echo "  ls -la test_result/"
echo "  cat logs/config1.log"
echo ""
echo "=========================================="
