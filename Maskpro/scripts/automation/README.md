# MaskDM 自动化实验系统使用说明

本文档提供MaskDM自动化实验系统的完整使用指南，包括脚本说明、配置详情和操作步骤。

## 📁 目录结构

```
scripts/automation/
├── README.md                    # 本说明文档
├── quick_start.sh              # 🚀 一键启动脚本 (推荐)
├── run_all_experiments.sh      # 主控脚本
├── gpu_monitor.sh              # GPU监控面板
├── auto_config1.sh             # 配置1: 快速验证
├── auto_config2.sh             # 配置2: 标准平衡  
├── auto_config3.sh             # 配置3: 最佳性能
├── auto_config4.sh             # 配置4: GPU优化
└── auto_config5.sh             # 配置5: 精细调优
```

## 🚀 快速开始

### 最简单的方式 (推荐)

```bash
cd /data/xay/MaskDM/Maskpro
bash scripts/automation/quick_start.sh
```

选择适合的模式：
- **模式1**: 快速验证 (~30分钟)
- **模式2**: 标准实验 (~2小时)
- **模式3**: 完整实验 (~8-12小时)

### 手动运行单个配置

```bash
# 运行配置1 (快速验证)
bash scripts/automation/auto_config1.sh

# 运行配置2 (标准平衡)
bash scripts/automation/auto_config2.sh

# 运行其他配置...
```

## 📊 配置详情

| 配置 | GPU | 用途 | 数据集大小 | 批次 | 学习率 | 预计时间 |
|------|-----|------|-----------|------|--------|----------|
| **Config 1** | cuda:2 | 快速验证 | 5,000 | 16 | 0.5 | ~30分钟 |
| **Config 2** | cuda:3 | 标准平衡 | 20,000 | 64 | 1.0 | ~2小时 |
| **Config 3** | cuda:2 | 最佳性能 | 50,000 | 32 | 2.0 | ~6小时 |
| **Config 4** | cuda:3 | GPU优化 | 25,000 | 128 | 1.5 | ~3小时 |
| **Config 5** | cuda:2 | 精细调优 | 30,000 | 32 | 0.8 | ~8小时 |

### GPU分配策略
- **GPU 2 (cuda:2)**: Config 1, 3, 5 (奇数配置)
- **GPU 3 (cuda:3)**: Config 2, 4 (偶数配置)

## 🔧 详细使用说明

### 1. 环境准备

确保以下条件满足：

```bash
# 检查GPU状态
nvidia-smi

# 检查预训练模型
ls -la ../pretrained/ddpm_ema_cifar10/

# 检查Python环境
python --version
```

### 2. 运行方式

#### 方式A: 一键启动 (推荐)

```bash
cd /data/xay/MaskDM/Maskpro
bash scripts/automation/quick_start.sh
```

#### 方式B: 交互式主控脚本

```bash
bash scripts/automation/run_all_experiments.sh
```

选项：
- `1`: 并行运行所有配置
- `2`: 串行运行所有配置  
- `3`: 选择性运行特定配置
- `4`: 仅运行快速验证

#### 方式C: 直接运行单个配置

```bash
# 后台运行
nohup bash scripts/automation/auto_config1.sh > logs/config1.log 2>&1 &

# 前台运行 (可以看到实时输出)
bash scripts/automation/auto_config1.sh
```

### 3. 监控实验进度

#### 启动监控面板

```bash
bash scripts/automation/gpu_monitor.sh
```

监控面板功能：
- 🖥️ 实时GPU状态 (利用率、内存、温度)
- 📊 实验进度跟踪
- 📁 结果统计
- 📝 日志查看
- ⏹️ 实验终止

#### 查看日志

```bash
# 实时跟踪配置1日志
tail -f logs/config1.log

# 查看所有日志
ls -la logs/

# 查看特定配置日志
less logs/config2.log
```

### 4. 实验流程

每个配置自动执行三个步骤：

```
步骤1: 生成基线损失
├── 调用 inference_loss_diffusion.py
├── 生成baseline loss文件
└── 保存到对应目录

步骤2: 训练MaskDM模型  
├── 调用 train_diffusion.py
├── 使用步骤1的损失文件
└── 保存到 train_result/

步骤3: 测试评估
├── 调用 test_ddpm_maskpro.py
├── 使用步骤2的模型
└── 保存到 test_result/
```

## 📁 结果管理

### 结果目录结构

```
MaskDM/Maskpro/
├── train_result/
│   ├── config1_conservative/
│   ├── config2_standard/
│   ├── config3_aggressive/
│   ├── config4_efficient/
│   └── config5_finetuning/
├── test_result/
│   ├── config1_conservative/
│   ├── config2_standard/
│   ├── config3_aggressive/
│   ├── config4_efficient/
│   └── config5_finetuning/
└── logs/
    ├── config1.log
    ├── config2.log
    ├── config3.log
    ├── config4.log
    └── config5.log
```

### 查看结果

```bash
# 列出所有训练结果
ls -la train_result/

# 列出所有测试结果  
ls -la test_result/

# 查看特定配置的训练结果
ls -la train_result/config1_conservative/

# 查看FID分数等测试指标
cat test_result/config1_conservative/metrics.json
```

## 🛠️ 高级操作

### 并行运行多个配置

```bash
# GPU2运行配置1和3
nohup bash scripts/automation/auto_config1.sh > logs/config1.log 2>&1 &
nohup bash scripts/automation/auto_config3.sh > logs/config3.log 2>&1 &

# GPU3运行配置2和4
nohup bash scripts/automation/auto_config2.sh > logs/config2.log 2>&1 &
nohup bash scripts/automation/auto_config4.sh > logs/config4.log 2>&1 &
```

### 终止实验

```bash
# 终止所有自动化实验
pkill -f auto_config

# 终止特定配置
pkill -f auto_config1.sh

# 查看运行中的实验
ps aux | grep auto_config
```

### 修改配置参数

直接编辑对应的脚本文件：

```bash
# 修改配置1参数
nano scripts/automation/auto_config1.sh

# 主要可修改参数：
# - GPU_ID: GPU编号
# - dataset_size: 数据集大小
# - batch_size: 批次大小
# - lr: 学习率
# - epoch: 训练轮数
# - logits: logits倍数
```

## 📋 常见问题

### Q1: GPU内存不足怎么办？

```bash
# 减小批次大小
# 在脚本中修改 --batch_size 参数
# 例如：从128改为64或32
```

### Q2: 如何只运行部分步骤？

```bash
# 只生成损失
python inference_loss_diffusion.py --original_model "../pretrained/ddpm_ema_cifar10" ...

# 只训练 (需要先有损失文件)
CUDA_VISIBLE_DEVICES=2 python train_diffusion.py --original_model ...

# 只测试 (需要先有训练好的模型)
python test_ddpm_maskpro.py --checkpoint_path "train_result/config1_conservative" ...
```

### Q3: 如何检查实验是否成功？

```bash
# 检查日志文件
grep "完成\|失败\|错误" logs/config1.log

# 检查结果目录
ls -la train_result/config1_conservative/
ls -la test_result/config1_conservative/

# 检查进程状态
ps aux | grep auto_config
```

### Q4: 如何自定义新配置？

1. 复制现有配置文件：
```bash
cp scripts/automation/auto_config1.sh scripts/automation/auto_config_custom.sh
```

2. 修改参数：
```bash
nano scripts/automation/auto_config_custom.sh
# 修改CONFIG_NAME, GPU_ID和各种超参数
```

3. 运行自定义配置：
```bash
bash scripts/automation/auto_config_custom.sh
```

## 🔍 性能建议

### 资源优化

- **Config 1**: 适合快速验证，资源消耗最小
- **Config 4**: 大批次，需要更多GPU内存但效率高
- **Config 3,5**: 完整数据集，需要更多存储空间

### 运行策略

1. **首次使用**: 先运行Config 1验证环境
2. **参数调优**: 使用Config 2进行初步探索
3. **最终实验**: 使用Config 3或5获得最佳结果
4. **效率优先**: 使用Config 4充分利用GPU

### 监控建议

- 使用 `gpu_monitor.sh` 实时监控GPU状态
- 定期检查日志文件避免错误累积
- 并行运行时注意GPU内存分配

## 📞 技术支持

如遇到问题，请检查：

1. **环境依赖**: Python环境、CUDA版本、依赖包
2. **文件路径**: 预训练模型、数据集路径是否正确
3. **权限设置**: 脚本是否有执行权限
4. **资源状态**: GPU内存、磁盘空间是否充足

---

*最后更新: $(date)*
*MaskDM自动化实验系统 v1.0*
