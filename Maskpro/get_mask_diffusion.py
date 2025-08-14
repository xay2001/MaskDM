import torch
import torch.nn as nn
from diffusers import DDPMPipeline
import os
import argparse

def mask_wrapper_diffusion(module, prefix=""):
    """
    Extract masks from pruned diffusion model (UNet)
    处理扩散模型的掩码提取，支持Conv2d和Linear层
    """
    
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        # 递归处理子模块
        mask_wrapper_diffusion(child, full_name)
    
    # 处理Conv2d和Linear层
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if hasattr(module, 'weight') and module.weight is not None:
            # 生成掩码：非零权重位置为True
            mask = module.weight.data != 0
            
            # 确保initial_mask目录存在
            os.makedirs("initial_mask", exist_ok=True)
            
            # 保存掩码
            save_path = f"initial_mask/{prefix}.pt"
            torch.save(mask, save_path)
            
            # 打印信息
            total_params = module.weight.numel()
            non_zero_params = torch.count_nonzero(module.weight).item()
            sparsity = (1 - non_zero_params / total_params) * 100
            
            print(f"Saved mask: {save_path}")
            print(f"  Shape: {mask.shape}")
            print(f"  Sparsity: {sparsity:.2f}% ({non_zero_params}/{total_params})")
            print()

def load_pruned_diffusion_model(model_path, device='cpu'):
    """
    加载剪枝后的扩散模型
    """
    try:
        # 方法1: 尝试直接加载保存的UNet模型
        unet_path = os.path.join(model_path, "pruned", "unet_weight_pruned.pth")
        if os.path.exists(unet_path):
            print(f"Loading pruned UNet from: {unet_path}")
            unet = torch.load(unet_path, map_location=device)
            return unet
        else:
            # 方法2: 尝试加载完整的pipeline
            print(f"Loading pipeline from: {model_path}")
            pipeline = DDPMPipeline.from_pretrained(model_path)
            return pipeline.unet
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract masks from pruned diffusion model")
    parser.add_argument("--model_path", type=str, 
                       default="/data/xay/MaskDM/run/pruned/weight_magnitude/ddpm_cifar10_weight_pruned",
                       help="Path to the pruned diffusion model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load model on")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Extracting masks from pruned diffusion model")
    print("="*60)
    
    # 加载剪枝后的模型
    print(f"Loading model from: {args.model_path}")
    model = load_pruned_diffusion_model(args.model_path, args.device)
    
    if model is None:
        print("Failed to load model!")
        exit(1)
    
    print(f"Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print()
    
    # 统计模型中的层数
    conv_count = 0
    linear_count = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv_count += 1
        elif isinstance(module, nn.Linear):
            linear_count += 1
    
    print(f"Found {conv_count} Conv2d layers and {linear_count} Linear layers")
    print()
    
    # 提取掩码
    print("Extracting masks...")
    mask_wrapper_diffusion(model)
    
    print("="*60)
    print("Mask extraction completed!")
    
    # 统计生成的掩码文件
    mask_files = [f for f in os.listdir("initial_mask") if f.endswith('.pt')]
    print(f"Generated {len(mask_files)} mask files in initial_mask/")
    print("="*60)
