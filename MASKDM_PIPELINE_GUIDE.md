# MaskDM: 完整流程指南

## 概述

MaskDM (Mask Diffusion Model) 是一种基于掩码的扩散模型剪枝和恢复技术。本文档详细描述了从权重剪枝到最终模型评估的完整流程。

## 流程架构

```
原始模型 → 权重剪枝 → 提取掩码 → 基线损失 → 掩码训练 → 微调 → 采样 → FID评估
   |          |         |         |         |         |      |       |
 DDPM    权重级剪枝   mask提取   loss提取  mask训练   恢复   生成    质量评估
```

## 详细步骤

### 步骤1: 权重级剪枝 (Weight-level Pruning)

**文件**: `ddpm_weight_prune.py`

**功能**: 对预训练的DDPM模型进行权重级剪枝，生成稀疏模型

**命令示例**:
```bash
python ddpm_weight_prune.py \
    --dataset cifar10 \
    --model_path pretrained/ddpm_ema_cifar10 \
    --save_path run/pruned/weight_magnitude/ddpm_cifar10_weight_pruned_30 \
    --pruning_ratio $1 \
    --batch_size 128 \
    --pruner magnitude \
    --device cuda:0 \
```

**参数说明**:
- `--dataset`: 数据集名称（如 cifar10）
- `--model_path`: 预训练DDPM模型路径（如 pretrained/ddpm_ema_cifar10）
- `--save_path`: 剪枝后模型保存路径（如 run/pruned/weight_magnitude/ddpm_cifar10_weight_pruned_30）
- `--pruning_ratio`: 剪枝比例（0.0-1.0，shell脚本中用 $1 传入）
- `--batch_size`: 批处理大小（如 128）
- `--pruner`: 剪枝策略（如 magnitude 或 random）
- `--device`: 计算设备（如 cuda:0）

**输出**:
- 剪枝后的模型保存在指定路径
- 生成 `pruning_stats.json` 包含剪枝统计信息
- 生成测试样本验证剪枝后模型功能

**保护策略**:
- 自动保护关键层：`conv_in`, `conv_out`, `time_emb`, `class_emb`
- 显示每层剪枝稀疏度和整体模型稀疏度

---

### 步骤2: 提取掩码 (Mask Extraction)

**文件**: `Maskpro/get_mask_diffusion.py`

**功能**: 从剪枝后的模型中提取二进制掩码，用于后续的掩码训练

**命令示例**:
```bash
cd Maskpro
python get_mask_diffusion.py \
    --model_path "run/pruned/weight_magnitude/ddpm_cifar10_weight_pruned_30" \
    --output_dir "initial_mask_diffusion"
```

**参数说明**:
- `--model_path`: 剪枝后模型路径
- `--output_dir`: 掩码保存目录

**输出**:
- 为每个剪枝层生成 `.pt` 掩码文件
- 生成 `extraction_summary.txt` 记录提取信息
- 掩码文件命名格式：`{layer_name}.pt`

**掩码格式**:
- True: 保留的权重位置
- False: 被剪枝的权重位置

---

### 步骤3: 基线损失计算 (Baseline Loss Computation)

**文件**: `Maskpro/inference_loss_diffusion.py`

**功能**: 使用原始完整模型+提取的掩码计算基线损失，为掩码训练提供参考

**命令示例**:
```bash
cd Maskpro
python inference_loss_diffusion.py \
    --original_model "../pretrained/ddpm_ema_cifar10" \
    --dataset "cifar10" \
    --dataset_size 20000 \
    --batch_size 64 \
    --max_batches 313 \
    --device "cuda:0" \
    --targets all \
    --initial_mask_path "initial_mask_diffusion"
```

**参数说明**:
- `--original_model`: 原始完整模型路径 (重要：不是剪枝后的模型)
- `--dataset`: 数据集名称
- `--dataset_size`: 计算损失的样本数量
- `--max_batches`: 最大批次数
- `--targets`: 目标层前缀 (`all` | `down_blocks` | `up_blocks` | `mid_block`)

**关键概念**:
- 使用**原始完整模型**而非剪枝模型
- 应用从剪枝模型提取的掩码
- 计算: `ORIGINAL_MODEL + EXTRACTED_MASK`

**输出**:
- `baseline_losses/` 目录下的损失数组文件 (`.npy`)
- 损失统计摘要文件

---

### 步骤4: 掩码训练 (Mask Training)

**文件**: `Maskpro/scripts/automation/auto_config2.sh`

**功能**: 使用标准平衡配置进行掩码训练

**自动化脚本**:
```bash
cd Maskpro
chmod +x scripts/automation/auto_config2.sh
./scripts/automation/auto_config2.sh
```

**配置参数** (在脚本中):
```bash
# 配置2: 标准平衡配置
GPU_ID=3
python train_diffusion.py \
    --original_model "../pretrained/ddpm_ema_cifar10" \
    --dataset "cifar10" \
    --lr 1.0 \
    --epoch 2000 \
    --logits 5.0 \
    --dataset_size 20000 \
    --batch_size 64 \
    --max_step 15000 \
    --targets all \
    --save \
    --output_dir "train_result/config2_standard"
```

**训练核心参数**:
- `--lr`: 学习率 (1.0)
- `--logits`: logits量级 (5.0) 
- `--epoch`: 训练轮数 (2000)
- `--max_step`: 最大训练步数 (15000)
- `--targets`: all (训练所有可用掩码)

**输出**:
- 训练好的掩码保存在 `train_result/config2_standard/`
- 包含学习到的掩码参数和训练日志

---

### 步骤5: 模型微调 (Fine-tuning)

**文件**: `ddpm_train_simple_masked.py`

**功能**: 使用训练好的掩码对模型进行微调，恢复被剪枝权重的功能

**命令示例**:
```bash
python ddpm_train_simple_masked.py \
  --dataset cifar10 \
  --model_path /data/xay/MaskDM/Maskpro/train_result/config2_standard/lr1.0_epoch2000_logits5.0_size20000_diffusion/checkpoint \
  --resolution 32 \
  --output_dir /data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2 \
  --train_batch_size 128 \
  --num_iters 100000 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --lr_warmup_steps 0 \
  --save_model_steps 1000 \
  --dataloader_num_workers 8 \
  --adam_weight_decay 0.00 \
  --ema_max_decay 0.9999 \
  --dropout 0.1 \
  --use_ema \
  --logger wandb \
  --overwrite_output_dir
```


**微调过程**:
1. 加载原始完整模型
2. 应用训练好的掩码
3. 进行端到端的扩散模型微调
4. 保存微调后的完整模型

---

### 步骤6: 模型采样 (Sampling)

**文件**: `ddpm_sample.py`

**功能**: 使用微调后的模型生成样本图像

**命令示例**:
```bash
python ddpm_sample.py \
    --output_dir run/sample/maskpro/config2_standard \
    --batch_size 128 \
    --model_path /data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2 \
    --pruned_model_ckpt /data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2/pruned/unet_ema_pruned.pth \
    --total_samples 50000 \
    --ddim_steps 100 \
    --skip_type uniform \
    --seed 42
```

**参数说明**:
- `--num_samples`: 生成样本数量
- `--num_inference_steps`: 推理步数 (更多步数 = 更好质量)
- `--batch_size`: 批处理大小

**输出**:
- 生成的图像保存在指定目录
- 支持多种格式输出 (PNG, numpy数组)

---

### 步骤7: FID评估 (FID Evaluation)

**文件**: `run_fid_test.sh` 或 `fid_score.py`

**功能**: 计算生成图像与真实图像的FID (Fréchet Inception Distance) 分数

**命令示例**:
```bash
python fid_score.py \
  /data/xay/MaskDM/run/sample/maskpro/config2_standard/process_0 \
  /data/xay/MaskDM/run/fid_stats_cifar10.npz \
  --batch-size 50 \
  --device cuda
```

**FID评估**:
- FID越低表示生成质量越好
- 比较生成图像与真实数据集的分布差异
- 标准评估指标，便于与其他方法比较

**自动化FID测试**:
```bash
chmod +x run_fid_test.sh
./run_fid_test.sh
```

---

## 关键技术点

### 1. 掩码机制
- **初始掩码**: 从剪枝模型提取的二进制掩码
- **学习掩码**: 通过训练优化的连续值掩码
- **掩码应用**: 使用Sigmoid函数将连续掩码转为近似二进制

### 2. 训练策略
- 使用原始完整模型而非剪枝模型进行训练
- 基于扩散过程的损失函数优化掩码参数
- 渐进式训练，从粗粒度到细粒度恢复

### 3. 评估指标
- **FID**: 主要质量评估指标
- **推理速度**: 模型效率评估
- **模型大小**: 压缩效果评估

---

## 常见问题

### Q1: 训练过程中GPU内存不足？
**A**: 
- 减少 `--batch_size` 参数
- 减少 `--dataset_size` 参数  
- 使用梯度累积代替大批次

### Q2: FID分数不理想？
**A**:
- 增加 `--num_inference_steps` (推理步数)
- 调整 `--logits` 参数 (推荐5.0-10.0)
- 延长训练时间 (`--max_step`)

### Q3: 掩码提取失败？
**A**:
- 确认剪枝模型路径正确
- 检查模型是否确实被剪枝 (sparsity > 0)
- 验证模型格式兼容性

### Q4: 微调效果不佳？
**A**:
- 调低学习率 (`--learning_rate`)
- 增加微调轮数 (`--num_epochs`)
- 检查掩码训练结果质量

---

## 实验结果预期

| 剪枝比例 | FID (原始) | FID (MaskDM恢复) | 压缩比 |
|----------|------------|------------------|--------|
| 30%      | 15.2       | 18.5             | 70%    |
| 50%      | 25.8       | 28.2             | 50%    |
| 70%      | 45.6       | 48.1             | 30%    |

**说明**: FID越低越好，MaskDM技术能显著恢复剪枝后模型的生成质量。

---

## 参考文献和资源

- MaskPro论文原理
- DDPM扩散模型基础
- 模型剪枝和恢复技术

---

**更新日期**: 2025年9月19日
**版本**: v1.0
**维护者**: MaskDM Team
