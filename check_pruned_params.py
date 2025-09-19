#!/usr/bin/env python3
"""
检查真正剪枝后模型的参数量
"""

import torch
import os

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_params(num_params):
    """格式化参数数量"""
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"
    else:
        return str(num_params)

def main():
    # 检查原始的带mask模型
    mask_model_path = "/data/xay/MaskDM/Maskpro/train_result/config2_standard/lr1.0_epoch2000_logits5.0_size20000_diffusion/checkpoint"
    
    # 检查真正剪枝后的模型
    pruned_model_path = "/data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2/pruned/unet_ema_pruned.pth"
    
    print("🔍 检查模型参数量对比...")
    
    if os.path.exists(pruned_model_path):
        print(f"\n📊 真正剪枝后的模型:")
        print(f"路径: {pruned_model_path}")
        
        try:
            # 加载剪枝后的模型
            pruned_model = torch.load(pruned_model_path, map_location='cpu')
            total_params, trainable_params = count_parameters(pruned_model)
            
            print(f"总参数量: {format_params(total_params)} ({total_params:,})")
            print(f"可训练参数: {format_params(trainable_params)} ({trainable_params:,})")
            
            # 显示一些层的信息
            print(f"\n🏗️ 模型类型: {type(pruned_model).__name__}")
            
            # 计算剪枝比例（假设原始模型是35.75M参数）
            original_params = 35746307
            pruning_ratio = (original_params - total_params) / original_params * 100
            print(f"📉 剪枝比例: {pruning_ratio:.2f}% (相比原始模型)")
            
        except Exception as e:
            print(f"❌ 加载剪枝模型失败: {e}")
    else:
        print(f"❌ 剪枝模型文件不存在: {pruned_model_path}")
    
    # 也检查一下其他可能的剪枝模型
    other_pruned_files = [
        "/data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2/pruned/unet_pruned.pth",
        "/data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2/pruned/unet_ema_pruned-100000.pth"
    ]
    
    for pruned_file in other_pruned_files:
        if os.path.exists(pruned_file):
            print(f"\n📊 检查文件: {os.path.basename(pruned_file)}")
            try:
                model = torch.load(pruned_file, map_location='cpu')
                total_params, _ = count_parameters(model)
                print(f"参数量: {format_params(total_params)} ({total_params:,})")
            except Exception as e:
                print(f"❌ 加载失败: {e}")

if __name__ == "__main__":
    main()


