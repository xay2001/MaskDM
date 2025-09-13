# DDPM MaskPro

该目录包含DDPM版本的MaskPro - 一个用于优化神经网络稀疏掩码的概率学习框架。原始MaskPro是为大语言模型（LLaMA-2-7B）设计的，此版本将其扩展到扩散模型（DDPM）。

## 📁 完整目录结构

```
Maskpro/
├── README_DDPM_中文.md           # 本说明文档
├── README.md                    # 英文简版说明
├── 
├── 🔧 核心实现文件
├── get_mask_diffusion.py        # 从剪枝的DDPM模型中提取二进制掩码
├── wrapper_diffusion.py         # DDPM模型的掩码包装器/解包器
├── inference_loss_diffusion.py  # 计算PGE的基线扩散损失
├── train_diffusion.py           # DDPM的主要MaskPro训练脚本
├── test_ddpm_maskpro.py          # 训练模型的评估脚本
├── 
├── 💼 传统LLM版本文件
├── get_mask.py                  # LLM版本掩码提取
├── wrapper.py                   # LLM版本包装器
├── inference_loss.py            # LLM版本基线损失
├── train.py                     # LLM版本训练脚本
├── 
├── 🚀 自动化脚本系统
├── scripts/
│   ├── automation/              # 一键启动脚本
│   │   ├── README.md           # 自动化系统详细说明
│   │   ├── quick_start.sh      # 🚀 一键启动脚本 (推荐)
│   │   ├── run_all_experiments.sh  # 主控脚本
│   │   ├── gpu_monitor.sh      # GPU监控面板
│   │   ├── auto_config1.sh     # 配置1: 快速验证
│   │   ├── auto_config2.sh     # 配置2: 标准平衡
│   │   ├── auto_config3.sh     # 配置3: 最佳性能
│   │   ├── auto_config4.sh     # 配置4: GPU优化
│   │   └── auto_config5.sh     # 配置5: 精细调优
│   ├── experiment_configs.md   # 配置详情对比
│   ├── loss/                   # 基线损失生成脚本
│   ├── train/                  # 训练脚本配置
│   └── test/                   # 测试评估脚本
├── 
├── 💾 数据和结果目录
├── initial_mask_diffusion/      # 从剪枝模型提取的初始掩码
├── learned_mask_diffusion/      # 学习得到的优化掩码
├── baseline_losses/             # 基线损失文件
├── results_diffusion/           # 训练结果
├── train_result/               # 按配置组织的训练结果
│   ├── config1_conservative/
│   ├── config2_standard/
│   ├── config3_aggressive/
│   ├── config4_efficient/
│   └── config5_finetuning/
├── test_result/                # 按配置组织的测试结果
│   └── (同上)
├── swanlog/                    # SwanLab监控日志
├── logs/                       # 运行日志
├── data_cache/                 # 数据缓存
├── model_cache/                # 模型缓存
└── __pycache__/                # Python缓存
```

## 🔥 概述

DDPM MaskPro应用与原始MaskPro相同的核心概念，但适配了扩散模型：

- **概率稀疏掩码优化** 用于Conv2d和Linear层
- **(N:M) 稀疏性模式** （默认：2:4稀疏性）  
- **精细化策略梯度估计 (PGE)** 用于噪声预测任务
- **权重级剪枝** 兼容扩散模型架构
- **SwanLab实验监控** 实时跟踪训练过程
- **自动化实验系统** 一键运行多种配置

## 🚀 快速开始 (推荐)

### 最简单的方式

```bash
cd /data/xay/MaskDM/Maskpro
bash scripts/automation/quick_start.sh
```

选择适合的模式：
- **模式1**: 🚀 快速验证 (~30分钟)
- **模式2**: ⚖️ 标准实验 (~2小时)
- **模式3**: 🔥 完整实验 (~8-12小时)
- **模式4**: 🎯 自定义选择
- **模式5**: 📊 仅启动监控面板

### GPU监控

```bash
bash scripts/automation/gpu_monitor.sh
```

实时监控功能：
- 🖥️ GPU状态 (利用率、内存、温度)
- 📊 实验进度跟踪
- 📁 结果统计
- 📝 日志查看
- ⏹️ 实验终止

## 📋 前置要求

### 1. 预训练模型准备

首先确保有完整的DDPM预训练模型：

```bash
# 检查预训练模型是否存在
ls -la ../pretrained/ddpm_ema_cifar10/

# 如果不存在，从HuggingFace转换
bash ../tools/convert_cifar10_ddpm_ema.sh
```

### 2. 剪枝模型准备

使用主仓库创建权重剪枝的DDPM模型：

```bash
cd ..
bash scripts/weight_prune/prune_ddpm_weight_magnitude_cifar10.sh 0.5
```

### 3. 环境要求

确保具有与主DDPM仓库相同的环境，额外安装：

```bash
pip install swanlab  # 可选：实验监控
pip install matplotlib scipy torchvision  # 可选：FID计算和可视化
```

## 🎯 五种配置详情

| 配置 | 用途 | 数据集大小 | 批次 | 学习率 | Logits | 预计时间 | GPU |
|------|------|-----------|------|--------|--------|----------|-----|
| **Config 1** | 快速验证 | 5,000 | 16 | 0.5 | 3.0 | ~30分钟 | cuda:2 |
| **Config 2** | 标准平衡 | 20,000 | 64 | 1.0 | 5.0 | ~2小时 | cuda:3 |
| **Config 3** | 最佳性能 | 50,000 | 32 | 2.0 | 7.0 | ~6小时 | cuda:2 |
| **Config 4** | GPU优化 | 25,000 | 128 | 1.5 | 6.0 | ~3小时 | cuda:3 |
| **Config 5** | 精细调优 | 30,000 | 32 | 0.8 | 10.0 | ~8小时 | cuda:2 |

详细配置说明请参考：`scripts/experiment_configs.md`

## 📖 手动分步执行

如需要手动控制每个步骤：

### 步骤1：提取掩码

```bash
python get_mask_diffusion.py \
    --model_path ../run/pruned/weight_magnitude/ddpm_cifar10_weight_pruned \
    --output_dir initial_mask_diffusion
```

### 步骤2：计算基线损失

```bash
python inference_loss_diffusion.py \
    --original_model "../pretrained/ddpm_ema_cifar10" \
    --dataset cifar10 \
    --dataset_size 512 \
    --batch_size 32 \
    --device cuda:0
```

### 步骤3：训练掩码优化

```bash
python train_diffusion.py \
    --original_model "../pretrained/ddpm_ema_cifar10" \
    --dataset cifar10 \
    --lr 50 \
    --epoch 625 \
    --logits 10.0 \
    --dataset_size 512 \
    --batch_size 32 \
    --targets down_blocks up_blocks mid_block \
    --save \
    --project_name "DDPM-MaskPro" \
    --output_dir results_diffusion
```

### 步骤4：测试评估

```bash
python test_ddpm_maskpro.py \
    --checkpoint_path results_diffusion/lr50_epoch625_logits10_size512_diffusion/checkpoint \
    --original_model "../pretrained/ddpm_ema_cifar10" \
    --num_samples 1000 \
    --compute_fid \
    --output_dir test_results
```

## 🔧 主要核心文件详解

### 核心实现文件

- **`get_mask_diffusion.py`** - 从剪枝的DDPM模型中提取二进制掩码
  - 自动跳过保护层 (`conv_in`, `conv_out`, `time_emb`, `class_emb`)
  - 支持Conv2d和Linear层
  - 只提取已剪枝层的掩码

- **`wrapper_diffusion.py`** - DDPM模型的掩码包装器和解包器
  - `mask_wrapper_diffusion()` - 为模型应用掩码功能
  - `mask_unwrapper_diffusion()` - 移除掩码包装并保存最终模型
  - `generate_mask()` - 基于logits生成(2:4)稀疏掩码
  - 支持Conv2d和Linear层的不同处理策略

- **`inference_loss_diffusion.py`** - 计算PGE的基线扩散损失
  - 计算：ORIGINAL_MODEL + INITIAL_MASK 的损失
  - 支持不同数据集大小和批次配置
  - 保存基线损失用于训练

- **`train_diffusion.py`** - DDPM的主要MaskPro训练脚本
  - 实现：ORIGINAL_MODEL + DYNAMIC_MASK 训练策略
  - 集成SwanLab实验监控
  - 支持多种超参数配置
  - 自动保存训练曲线和统计信息

- **`test_ddpm_maskpro.py`** - 训练模型的评估脚本
  - 对比：ORIGINAL_MODEL + LEARNED_MASK vs ORIGINAL_MODEL + INITIAL_MASK
  - 生成样本图像和网格
  - 计算FID分数 (如果安装了scipy)
  - 评估重建损失和模型统计

### 自动化脚本系统

- **`scripts/automation/quick_start.sh`** - 🚀 **推荐使用** 的一键启动脚本
- **`scripts/automation/run_all_experiments.sh`** - 交互式主控脚本
- **`scripts/automation/gpu_monitor.sh`** - 实时GPU监控面板
- **`scripts/automation/auto_config*.sh`** - 5种不同的自动化配置
- **`scripts/automation/README.md`** - 详细的自动化系统说明

## 🎛️ 训练参数详解

### 基本参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--original_model` | `../pretrained/ddpm_ema_cifar10` | **重要**: 原始完整模型路径 (非剪枝模型) |
| `--dataset` | `cifar10` | 数据集名称 |
| `--lr` | `50` | logits学习率 (关键超参数) |
| `--epoch` | `625` | 训练轮数 (实际是steps/16) |
| `--logits` | `10.0` | 初始logits倍数 |
| `--dataset_size` | `512` | 训练样本数量 |
| `--batch_size` | `32` | 批次大小 |
| `--max_step` | `10000` | 最大训练步数 |

### 目标层选择

| 参数值 | 说明 |
|--------|------|
| `--targets all` | 优化所有有掩码的层 |
| `--targets down_blocks` | 只优化下采样块 |
| `--targets up_blocks` | 只优化上采样块 |  
| `--targets mid_block` | 只优化中间块 |
| `--targets down_blocks.1 up_blocks.2` | 优化特定子块 |

### SwanLab监控参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--project_name` | `DDPM-MaskPro` | SwanLab项目名称 |
| `--experiment_name` | 自动生成 | 实验名称 |
| `--disable_swanlab` | False | 禁用SwanLab监控 |

## 📊 输出结构

训练完成后的结果结构：

```
results_diffusion/lr50_epoch625_logits10_size512_diffusion/
├── checkpoint/                    # 优化的DDPM模型 (用于推理)
│   ├── model_index.json          # 模型配置
│   ├── unet/                     # UNet权重
│   └── scheduler/                # 调度器配置
├── logits/                       # 保存的logits (如果启用save)
├── loss_improvements.npy         # 训练改进曲线
├── loss_training.npy            # 训练损失曲线  
├── training_summary.json        # 训练统计信息
└── training_curves.png          # 训练曲线可视化 (如果有SwanLab)
```

## 🧪 评估和测试

### 基本测试

```bash
python test_ddpm_maskpro.py \
    --checkpoint_path results_diffusion/lr50_epoch625_logits10_size512_diffusion/checkpoint \
    --original_model "../pretrained/ddpm_ema_cifar10" \
    --num_samples 100
```

### 包含FID评估

```bash
python test_ddpm_maskpro.py \
    --checkpoint_path results_diffusion/lr50_epoch625_logits10_size512_diffusion/checkpoint \
    --original_model "../pretrained/ddpm_ema_cifar10" \
    --num_samples 1000 \
    --compute_fid \
    --output_dir test_results
```

### 测试结果

测试将生成：
- 📸 原始和优化模型的样本图像对比
- 📈 损失比较 (ORIGINAL+LEARNED vs ORIGINAL+INITIAL)
- 📊 模型统计 (稀疏性、参数数量)
- 🎯 性能指标 (重建损失、FID分数)
- 📋 详细评估报告 (`corrected_evaluation_results.json`)

## 🔬 与原始MaskPro的主要区别

### 架构适配

1. **层支持**：扩展支持DDPM中的Conv2d和Linear层
2. **保护层**：自动保护关键DDPM层：
   - `conv_in`, `conv_out` （输入/输出卷积）
   - `time_emb` （时间嵌入层）
   - `class_emb` （类别嵌入层）

3. **稀疏性模式**：为不同张量形状适配(N:M)稀疏性：
   - Conv2d：应用于展平的空间维度
   - Linear：直接应用于权重矩阵

### 损失函数和数据

- **原始MaskPro**：语言建模的交叉熵损失，Token序列输入
- **DDPM版本**：扩散过程中噪声预测的MSE损失，噪声图像+时间步输入
- **数据集**：CIFAR-10图像（而非C4文本）
- **预处理**：图像归一化到[-1, 1]范围

### 训练策略

- **基线计算**：ORIGINAL_MODEL + INITIAL_MASK 
- **训练目标**：ORIGINAL_MODEL + DYNAMIC_MASK
- **最终评估**：ORIGINAL_MODEL + LEARNED_MASK vs ORIGINAL_MODEL + INITIAL_MASK

## 🚀 高级功能

### 1. 实验监控

集成SwanLab进行全面监控：
- 📈 训练损失和改进曲线
- 🔍 模型稀疏性统计
- ⏱️ 性能计时分析
- 🎯 Logits分布监控
- 📊 自动生成训练报告

### 2. 多GPU并行

```bash
# 同时运行多个配置
nohup bash scripts/automation/auto_config1.sh > logs/config1.log 2>&1 &  # GPU 2
nohup bash scripts/automation/auto_config2.sh > logs/config2.log 2>&1 &  # GPU 3
```

### 3. 自定义目标层

```bash
python train_diffusion.py \
    --targets down_blocks.1.attentions.0 up_blocks.2.resnets.1 \
    --lr 30 \
    [其他参数...]
```

### 4. 批量实验管理

```bash
# 启动所有配置的并行实验
bash scripts/automation/run_all_experiments.sh

# 选项1: 并行运行所有配置 (需要多GPU)
# 选项2: 串行运行所有配置 (单GPU)
# 选项3: 选择性运行特定配置
# 选项4: 仅运行快速验证
```

## 🛠️ 超参数调优指南

### 学习率调优
- **lr=20-50**: 稳定收敛，适合初始实验
- **lr=50-100**: 更快收敛，可能需要监控稳定性
- **lr>100**: 激进设置，需要仔细调试

### Logits倍数调优
- **logits=3-5**: 保守初始化，稳定但可能收敛慢
- **logits=10**: 标准设置，平衡效果和稳定性
- **logits=15+**: 激进初始化，可能收敛快但不稳定

### 数据集大小权衡
- **dataset_size=512-5000**: 快速迭代和调试
- **dataset_size=10000-25000**: 平衡质量和速度
- **dataset_size=50000**: 最佳质量，但需要更长时间

### 批次大小选择
- **batch_size=16-32**: 内存友好，适合大多数GPU
- **batch_size=64-128**: 需要更多GPU内存，但可能更稳定
- **动态调整**: 如果遇到OOM，逐步减小批次大小

## 🔧 故障排除

### 常见问题及解决方案

#### 1. "未找到初始掩码"
```bash
# 确保先运行了权重剪枝
cd ..
bash scripts/weight_prune/prune_ddpm_weight_magnitude_cifar10.sh 0.5

# 然后提取掩码
cd Maskpro
python get_mask_diffusion.py --model_path [剪枝模型路径]
```

#### 2. "未找到基线损失文件"
```bash
# 确保先计算了基线损失
python inference_loss_diffusion.py --dataset_size 512 --batch_size 32
```

#### 3. CUDA内存不足
```bash
# 减少批次大小
--batch_size 16  # 或更小

# 减少数据集大小
--dataset_size 256  # 或更小

# 使用梯度累积
--batch_size 8 --gradient_accumulation_steps 4
```

#### 4. 收敛问题
```bash
# 尝试不同学习率
--lr 20    # 更保守
--lr 30    # 中等
--lr 100   # 激进

# 调整logits初始化
--logits 5.0   # 更保守
--logits 15.0  # 更激进
```

### 调试模式

运行单个步骤进行调试：

```bash
# 检查掩码提取
python get_mask_diffusion.py \
    --model_path [路径] \
    --output_dir debug_masks

# 验证基线计算  
python inference_loss_diffusion.py \
    --dataset_size 64 \
    --max_batches 5

# 短训练测试
python train_diffusion.py \
    --epoch 10 \
    --dataset_size 64 \
    --batch_size 8
```

### 日志分析

```bash
# 查看训练进度
tail -f logs/config1.log

# 搜索错误信息
grep -i "error\|失败\|exception" logs/config1.log

# 检查GPU使用情况
nvidia-smi
```

## 📈 性能预期

### 典型结果 (CIFAR-10, 50%权重剪枝)

| 指标 | 预期值 |
|------|--------|
| **损失改进** | 0.1-0.5% MSE损失减少 |
| **训练时间** | 单GPU上625轮约30分钟 |
| **内存使用** | 与原始模型相似 |
| **收敛速度** | 前100轮内可见改进 |
| **最终稀疏性** | 保持原始剪枝比例 |

### 配置性能对比

- **训练速度**: Config 1 > Config 4 > Config 2 > Config 5 > Config 3
- **内存使用**: Config 4 > Config 2 > Config 3,5 > Config 1  
- **预期质量**: Config 5 > Config 3 > Config 2 > Config 4 > Config 1
- **稳定性**: Config 1,5 > Config 2 > Config 4 > Config 3

## 📚 引用

如果您使用此DDPM MaskPro适配版本，请引用原始MaskPro论文：

```bibtex
@article{sun2025maskpro,
  title={MaskPro: Linear-Space Probabilistic Learning for Strict (N: M)-Sparsity on Large Language Models},
  author={Sun, Yan and Zhang, Qixin and Yu, Zhiyuan and Zhang, Xikun and Shen, Li and Tao, Dacheng},
  journal={arXiv preprint arXiv:2506.12876},
  year={2025}
}

@inproceedings{fang2023structural,
  title={Structural pruning for diffusion models},
  author={Gongfan Fang and Xinyin Ma and Xinchao Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023},
}
```

## 🎯 快速命令参考

### 一键启动 (推荐)
```bash
bash scripts/automation/quick_start.sh
```

### GPU监控
```bash
bash scripts/automation/gpu_monitor.sh
```

### 手动运行完整流水线
```bash
# 1. 提取掩码
python get_mask_diffusion.py --model_path [剪枝模型路径]

# 2. 计算基线
python inference_loss_diffusion.py --dataset_size 512

# 3. 训练优化
python train_diffusion.py --lr 50 --epoch 625 --save

# 4. 测试评估  
python test_ddpm_maskpro.py --checkpoint_path [结果路径] --compute_fid
```

### 紧急停止
```bash
pkill -f auto_config    # 停止所有自动化实验
pkill -f train_diffusion    # 停止训练进程
pkill -f test_ddpm    # 停止测试进程
```

---

*最后更新: 2025年9月13  
*DDPM MaskPro v2.0 - 包含完整自动化系统和SwanLab监控*