#!/usr/bin/env python3
"""
检查mask vs 真正剪枝的差异
"""

import torch
import os

def analyze_model_structure(model_path, model_name):
    """分析模型结构和参数"""
    print(f"\n🔍 分析 {model_name}:")
    print(f"路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 文件不存在")
        return
    
    try:
        # 加载模型
        model = torch.load(model_path, map_location='cpu')
        
        # 统计参数
        total_params = 0
        zero_params = 0
        mask_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            
            # 检查是否有mask
            if 'mask' in name:
                mask_params += param.numel()
                print(f"  发现mask参数: {name} - {param.shape}")
            
            # 检查零参数
            if torch.sum(param == 0).item() > 0:
                zero_count = torch.sum(param == 0).item()
                zero_ratio = zero_count / param.numel() * 100
                if zero_ratio > 10:  # 超过10%的零参数才报告
                    print(f"  零参数层: {name} - {zero_ratio:.1f}% 为零")
                    zero_params += zero_count
        
        print(f"📊 总参数: {total_params:,}")
        print(f"📊 Mask参数: {mask_params:,}")
        print(f"📊 零参数: {zero_params:,}")
        print(f"📊 有效参数: {total_params - zero_params:,}")
        
        if zero_params > 0:
            effective_ratio = (total_params - zero_params) / total_params * 100
            print(f"📊 有效参数比例: {effective_ratio:.2f}%")
        
        return total_params, zero_params, mask_params
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None, None, None

def main():
    print("🔍 检查MaskPro的剪枝实现...")
    
    # 检查不同的模型文件
    models_to_check = [
        ("/data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2/pruned/unet_ema_pruned.pth", "微调后EMA剪枝模型"),
        ("/data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2/pruned/unet_pruned.pth", "微调后普通剪枝模型"),
    ]
    
    for model_path, model_name in models_to_check:
        analyze_model_structure(model_path, model_name)
    
    print("\n" + "="*60)
    print("🤔 结论分析:")
    print("如果看到大量零参数，说明是通过置零实现的'软剪枝'")
    print("如果参数总数不变，说明可能只是mask-based方法")
    print("真正的结构化剪枝应该会减少参数总数")

if __name__ == "__main__":
    main()


