#!/bin/bash

# 使用mask约束的剪枝模型微调脚本

echo "🚀 开始mask约束的剪枝模型微调..."

# 设置路径
PRUNED_MODEL_PATH="/data/xay/MaskDM/Maskpro/train_result/config2_standard/lr1.0_epoch2000_logits5.0_size20000_diffusion/checkpoint"
OUTPUT_DIR="/data/xay/MaskDM/finetuned_results/masked_finetuned"

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "📁 剪枝模型路径: $PRUNED_MODEL_PATH"
echo "📁 输出目录: $OUTPUT_DIR"

# 使用简化版mask约束微调脚本 - 避免序列化问题
python ddpm_train_simple_masked.py \
    --model_path $PRUNED_MODEL_PATH \
    --dataset cifar10 \
    --output_dir $OUTPUT_DIR \
    --resolution 32 \
    --train_batch_size 128 \
    --eval_batch_size 16 \
    --num_iters 50000 \
    --learning_rate 2e-4 \
    --lr_warmup_steps 0 \
    --use_ema \
    --ema_max_decay 0.9999 \
    --save_model_steps 1000 \
    --dataloader_num_workers 8 \
    --adam_weight_decay 0.00 \
    --dropout 0.1 \
    --gradient_accumulation_steps 1 \
    --mixed_precision fp16 \
    --logger wandb \
    --overwrite_output_dir

echo "✅ 微调完成！"
echo "📊 输出目录: $OUTPUT_DIR"
echo "🖼️  生成样本: $OUTPUT_DIR/vis/"
echo "💾 模型检查点: $OUTPUT_DIR/pruned/"

# 显示mask约束验证
echo ""
echo "🔍 验证mask约束是否正确保持..."
python -c "
import torch
import os
from safetensors.torch import load_file

output_dir = '$OUTPUT_DIR'
if os.path.exists(os.path.join(output_dir, 'pruned', 'unet_pruned.pth')):
    model = torch.load(os.path.join(output_dir, 'pruned', 'unet_pruned.pth'), map_location='cpu')
    
    mask_preserved = True
    for name, module in model.named_modules():
        if hasattr(module, 'mask') and hasattr(module, 'weight'):
            # 检查被mask的位置权重是否为0
            masked_positions = module.mask == 0
            weights_at_masked_positions = module.weight.data[masked_positions]
            if not torch.allclose(weights_at_masked_positions, torch.zeros_like(weights_at_masked_positions), atol=1e-6):
                print(f'❌ 层 {name} 的mask约束被违反！')
                mask_preserved = False
    
    if mask_preserved:
        print('✅ 所有mask约束都正确保持！')
    else:
        print('❌ 部分mask约束被违反！')
else:
    print('⚠️  尚未生成模型文件')
"
