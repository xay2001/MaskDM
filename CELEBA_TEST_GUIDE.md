# CelebA-HQ 数据集下载和测试完整指南

## 第一步：激活下载功能

在下载之前，需要手动激活CelebA数据集的下载功能：

```bash
# 编辑CelebA数据集文件
vim ddpm_exp/datasets/celeba.py
```

**需要进行的修改**：
1. **第65-66行**：取消注释下载代码
   ```python
   # 修改前：
   #if download:
   #    self.download()

   # 修改后：
   if download:
       self.download()
   ```

2. **第68-70行**：取消注释完整性检查
   ```python
   # 修改前：
   #if not self._check_integrity():
   #    raise RuntimeError('Dataset not found or corrupted.' +
   #                       ' You can use download=True to download it')

   # 修改后：
   if not self._check_integrity():
       raise RuntimeError('Dataset not found or corrupted.' +
                          ' You can use download=True to download it')
   ```

## 第二步：下载CelebA数据集

### 方法1: 使用自动下载脚本
```bash
# 使用项目提供的下载脚本
python download_celeba.py --data_root ./data/celeba
```

### 方法2: 手动下载
```bash
# 创建数据目录
mkdir -p data/celeba

# 进入ddpm_exp目录
cd ddpm_exp

# 运行Python下载
python -c "
from datasets.celeba import CelebA
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(64),
    transforms.ToTensor(),
])

dataset = CelebA(
    root='../data/celeba',
    split='train',
    download=True,
    transform=transform
)
print(f'CelebA数据集下载完成，共{len(dataset)}张训练图片')
"
```

## 第三步：验证数据集下载

```bash
# 检查数据集结构
ls -la data/celeba/
# 应该看到：
# - Img/          (图像文件夹)
# - Anno/         (标注文件夹) 
# - Eval/         (评估文件夹)

# 检查图像文件
ls data/celeba/Img/img_align_celeba/ | head -10
# 应该看到类似：000001.jpg, 000002.jpg 等文件
```

## 第四步：按照MaskDM流程进行测试

### 4.1 基础预训练模型测试
```bash
cd ddpm_exp

# 使用预训练模型生成样本
bash scripts/sample_celeba_pretrained.sh
```

### 4.2 完整剪枝和恢复流程

#### 步骤1: 权重剪枝
```bash
# 进入项目根目录
cd /data/xay/MaskDM

# 对CelebA模型进行权重剪枝（30%剪枝率）
python ddpm_weight_prune.py \
    --dataset celeba \
    --model_path pretrained/ddpm_ema_celeba \
    --save_path run/pruned/weight_magnitude/ddpm_celeba_weight_pruned_30 \
    --pruning_ratio 0.3 \
    --batch_size 64 \
    --pruner magnitude \
    --device cuda:0
```

#### 步骤2: 提取掩码
```bash
cd Maskpro

python get_mask_diffusion.py \
    --model_path "../run/pruned/weight_magnitude/ddpm_celeba_weight_pruned_30" \
    --output_dir "initial_mask_diffusion_celeba"
```

#### 步骤3: 基线损失计算
```bash
python inference_loss_diffusion.py \
    --original_model "../pretrained/ddpm_ema_celeba" \
    --dataset "celeba" \
    --dataset_size 10000 \
    --batch_size 32 \
    --max_batches 313 \
    --device "cuda:0" \
    --targets all \
    --initial_mask_path "initial_mask_diffusion_celeba"
```

#### 步骤4: 掩码训练
```bash
# 使用自动化配置进行掩码训练（需要修改配置）
python train_diffusion.py \
    --original_model "../pretrained/ddpm_ema_celeba" \
    --dataset "celeba" \
    --lr 1.0 \
    --epoch 2000 \
    --logits 5.0 \
    --dataset_size 10000 \
    --batch_size 32 \
    --max_step 15000 \
    --targets all \
    --save \
    --output_dir "train_result/celeba_config_standard"
```

#### 步骤5: 模型微调
```bash
cd ..  # 回到项目根目录

python ddpm_train_simple_masked.py \
    --dataset celeba \
    --model_path /data/xay/MaskDM/Maskpro/train_result/celeba_config_standard/lr1.0_epoch2000_logits5.0_size10000_diffusion/checkpoint \
    --resolution 64 \
    --output_dir /data/xay/MaskDM/finetuned_results/celeba_masked_finetuned \
    --train_batch_size 64 \
    --num_iters 50000 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-4 \
    --lr_warmup_steps 0 \
    --save_model_steps 1000 \
    --dataloader_num_workers 4 \
    --adam_weight_decay 0.00 \
    --ema_max_decay 0.9999 \
    --dropout 0.1 \
    --use_ema \
    --logger wandb \
    --overwrite_output_dir
```

#### 步骤6: 模型采样
```bash
python ddpm_sample.py \
    --output_dir run/sample/maskpro/celeba_standard \
    --batch_size 64 \
    --model_path /data/xay/MaskDM/finetuned_results/celeba_masked_finetuned \
    --pruned_model_ckpt /data/xay/MaskDM/finetuned_results/celeba_masked_finetuned/pruned/unet_ema_pruned.pth \
    --total_samples 10000 \
    --ddim_steps 100 \
    --skip_type uniform \
    --seed 42
```

#### 步骤7: FID评估
```bash
# 首先准备CelebA的FID统计数据（如果没有的话）
python fid_score.py \
    --compute-stats \
    data/celeba/Img/img_align_celeba \
    run/fid_stats_celeba.npz \
    --device cuda

# 计算FID分数
python fid_score.py \
    run/sample/maskpro/celeba_standard/process_0 \
    run/fid_stats_celeba.npz \
    --batch-size 50 \
    --device cuda
```

## 第五步：快速测试流程

### 使用现有脚本进行快速测试
```bash
cd ddmp_exp

# 完整的剪枝-微调-采样流程
bash scripts/run_celeba.sh 0.1  # 0.1为剪枝比例参数

# 或者分步执行：
# 1. 剪枝
bash scripts/prune_celeba_ddpm.sh

# 2. 微调
bash scripts/finetune_celeba_ddpm.sh  

# 3. 采样
bash scripts/sample_celeba_ddpm_pruning.sh
```

## 重要提示

### CelebA数据集特点：
- **图像尺寸**: 原始178×218，通常裁剪为64×64或128×128
- **数据量**: 约20万张人脸图像  
- **下载大小**: 约1.3GB压缩包
- **下载时间**: 取决于网络速度，通常需要10-30分钟

### 计算资源要求：
- **GPU内存**: 建议至少8GB VRAM
- **系统内存**: 建议至少16GB RAM  
- **存储空间**: 至少5GB可用空间
- **训练时间**: 完整流程约需要6-12小时

### 常见问题：
1. **下载失败**: 检查网络连接，可能需要多次尝试
2. **GPU内存不足**: 减小batch_size参数
3. **训练时间过长**: 可以减少max_step或epoch参数进行快速测试

### 质量评估标准：
- **原始模型 FID**: ~10-15 (CelebA 64×64)
- **30%剪枝后**: FID通常升高到20-30
- **MaskDM恢复后**: 目标是接近原始FID水平

这个完整流程展示了MaskDM技术在CelebA数据集上的剪枝和恢复能力。
