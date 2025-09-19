#!/usr/bin/env python3
"""
检查模型参数量的脚本
"""

import torch
from diffusers import UNet2DModel
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
    model_path = "/data/xay/MaskDM/Maskpro/train_result/config2_standard/lr1.0_epoch2000_logits5.0_size20000_diffusion/checkpoint"
    
    print("🔍 检查剪枝后模型参数量...")
    print(f"模型路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    try:
        # 加载UNet模型
        unet = UNet2DModel.from_pretrained(
            model_path, 
            subfolder="unet",
            low_cpu_mem_usage=False
        )
        
        # 计算参数量
        total_params, trainable_params = count_parameters(unet)
        
        print("\n📊 模型参数统计:")
        print(f"总参数量: {format_params(total_params)} ({total_params:,})")
        print(f"可训练参数: {format_params(trainable_params)} ({trainable_params:,})")
        
        # 显示模型结构概览
        print(f"\n🏗️ 模型结构:")
        print(f"模型类型: {type(unet).__name__}")
        print(f"输入通道: {unet.config.in_channels}")
        print(f"输出通道: {unet.config.out_channels}")
        print(f"Block通道: {unet.config.block_out_channels}")
        
        # 检查各层参数量
        print(f"\n📋 各模块参数量:")
        for name, module in unet.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                print(f"  {name}: {format_params(module_params)} ({module_params:,})")
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")

if __name__ == "__main__":
    main()


