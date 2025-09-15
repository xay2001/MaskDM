#!/usr/bin/env python3
"""
检查模型是否已经进行了mask剪枝
"""
import torch
import numpy as np
from diffusers import UNet2DModel
import os

def check_model_pruning(model_path):
    """检查模型是否已经进行了剪枝"""
    print(f"正在检查模型: {model_path}")
    
    # 加载模型
    if model_path.endswith('.pth') or model_path.endswith('.pt'):
        # 直接加载torch模型
        model = torch.load(model_path, map_location='cpu')
    else:
        # 从pretrained加载
        model = UNet2DModel.from_pretrained(model_path)
    
    total_params = 0
    zero_params = 0
    layer_stats = []
    
    print("\n=== 模型剪枝分析 ===")
    
    for name, param in model.named_parameters():
        param_data = param.data.cpu().numpy()
        total_count = param_data.size
        zero_count = np.sum(param_data == 0)
        
        total_params += total_count
        zero_params += zero_count
        
        sparsity = zero_count / total_count * 100
        
        layer_stats.append({
            'name': name,
            'total': total_count,
            'zeros': zero_count,
            'sparsity': sparsity
        })
        
        if sparsity > 1.0:  # 只显示稀疏度>1%的层
            print(f"{name}: {sparsity:.2f}% 稀疏度 ({zero_count}/{total_count})")
    
    overall_sparsity = zero_params / total_params * 100
    print(f"\n整体稀疏度: {overall_sparsity:.2f}% ({zero_params}/{total_params})")
    
    # 判断是否已剪枝
    if overall_sparsity > 5.0:
        print("✓ 模型已经进行了剪枝处理")
        return True
    else:
        print("✗ 模型可能未进行剪枝处理或剪枝程度很低")
        return False

def compare_with_original(original_path, pruned_path):
    """比较原始模型和剪枝模型"""
    print(f"\n=== 对比原始模型和剪枝模型 ===")
    
    # 加载原始模型
    if os.path.exists(original_path):
        original_model = UNet2DModel.from_pretrained(original_path)
        original_params = sum(p.numel() for p in original_model.parameters())
        print(f"原始模型参数量: {original_params:,}")
    
    # 检查剪枝模型
    is_pruned = check_model_pruning(pruned_path)
    
    return is_pruned

if __name__ == "__main__":
    # 检查剪枝模型
    pruned_model_path = "/data/xay/MaskDM/Maskpro/train_result/config2_standard/lr1.0_epoch2000_logits5.0_size20000_diffusion/checkpoint"
    
    print("正在检查剪枝模型...")
    is_pruned = check_model_pruning(pruned_model_path)
    
    # 也检查原始模型作为对比
    original_model_path = "/data/xay/MaskDM/pretrained/ddpm_ema_cifar10/unet"
    if os.path.exists(original_model_path):
        print("\n正在检查原始模型...")
        check_model_pruning(original_model_path)
    
    if is_pruned:
        print("\n✓ 建议：模型已经进行了剪枝，可以直接进行微调来恢复性能")
    else:
        print("\n⚠ 建议：如果模型应该是剪枝的，请检查剪枝过程是否正确执行")
