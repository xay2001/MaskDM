from diffusers import DiffusionPipeline, DDPMPipeline, DDIMPipeline, DDIMScheduler, DDPMScheduler
from diffusers.models import UNet2DModel
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
from torchvision import transforms
import torchvision
from tqdm import tqdm
import os
from glob import glob
from PIL import Image
import accelerate
import utils
import torch_pruning as tp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,  default=None, help="path to an image folder")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--pruning_ratio", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--pruner", type=str, default='magnitude', choices=['magnitude', 'random'])

args = parser.parse_args()

batch_size = args.batch_size
dataset = args.dataset

def count_parameters(model):
    """Count total and non-zero parameters"""
    total_params = 0
    non_zero_params = 0
    
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight'):
                # 如果有剪枝mask，统计剪枝后的参数
                if hasattr(module, 'weight_mask'):
                    weight = module.weight * module.weight_mask
                else:
                    weight = module.weight
                total_params += module.weight.numel()
                non_zero_params += torch.count_nonzero(weight).item()
            if hasattr(module, 'bias') and module.bias is not None:
                total_params += module.bias.numel()
                non_zero_params += torch.count_nonzero(module.bias).item()
    
    return total_params, non_zero_params

def calculate_real_macs(model, example_inputs):
    """Calculate real MACs by measuring actual operations with sparse weights"""
    total_macs = 0
    
    def add_hooks(module):
        def hook(module, input, output):
            nonlocal total_macs
            if isinstance(module, nn.Conv2d):
                if hasattr(module, 'weight'):
                    # 计算稀疏权重的比例
                    total_weights = module.weight.numel()
                    non_zero_weights = torch.count_nonzero(module.weight).item()
                    sparsity_ratio = non_zero_weights / total_weights if total_weights > 0 else 0
                    
                    # 计算输出特征图大小（除去batch维度）
                    output_elements = output.numel() // output.shape[0]
                    
                    # 每个输出位置需要的原始操作数 = input_channels * kernel_h * kernel_w
                    kernel_ops_per_output = module.weight.shape[1] * module.weight.shape[2] * module.weight.shape[3]
                    
                    # 稀疏权重下的实际操作数 = 输出位置数 * 每个位置的操作数 * 稀疏比例
                    actual_macs = output_elements * kernel_ops_per_output * sparsity_ratio
                    total_macs += actual_macs
                    
            elif isinstance(module, nn.Linear):
                if hasattr(module, 'weight'):
                    # Linear层：非零权重数量 * batch_size
                    non_zero_weights = torch.count_nonzero(module.weight).item()
                    batch_size = input[0].shape[0] if len(input) > 0 else 1
                    total_macs += non_zero_weights * batch_size
        
        return hook
    
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(add_hooks(module)))
    
    with torch.no_grad():
        model(**example_inputs)
    
    for hook in hooks:
        hook.remove()
    
    return total_macs

def apply_weight_pruning(model, pruning_ratio, method='magnitude'):
    """Apply weight-level pruning to Conv2d and Linear layers"""
    
    # Collect all Conv2d and Linear modules
    modules_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            modules_to_prune.append((module, 'weight'))
            
    print(f"Found {len(modules_to_prune)} layers to prune (Conv2d + Linear)")
    
    # Apply pruning based on method
    if method == 'magnitude':
        # Use L1Unstructured (magnitude-based) pruning
        for module, param_name in modules_to_prune:
            prune.l1_unstructured(module, name=param_name, amount=pruning_ratio)
    elif method == 'random':
        # Use RandomUnstructured pruning
        for module, param_name in modules_to_prune:
            prune.random_unstructured(module, name=param_name, amount=pruning_ratio)
    else:
        raise ValueError(f"Unknown pruning method: {method}")
    
    print(f"Applied {method} pruning with ratio {pruning_ratio}")

def remove_pruning_masks(model):
    """Remove pruning masks and make pruning permanent"""
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except:
                pass  # No pruning mask to remove

if __name__=='__main__':
    
    # Loading pretrained model
    print("Loading pretrained model from {}".format(args.model_path))
    pipeline = DDPMPipeline.from_pretrained(args.model_path).to(args.device)
    scheduler = pipeline.scheduler
    model = pipeline.unet.eval()
    
    if 'cifar' in args.model_path:
        example_inputs = {'sample': torch.randn(1, 3, 32, 32).to(args.device), 'timestep': torch.ones((1,)).long().to(args.device)}
    else:
        example_inputs = {'sample': torch.randn(1, 3, 256, 256).to(args.device), 'timestep': torch.ones((1,)).long().to(args.device)}

    if args.pruning_ratio > 0:
        # Count parameters and operations before pruning using torch_pruning (standard)
        base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
        
        print(f"Applying weight-level {args.pruner} pruning...")
        
        # Apply weight-level pruning
        apply_weight_pruning(model, args.pruning_ratio, args.pruner)
        
        # Count parameters after pruning (with masks)
        total_params, params = count_parameters(model)
        
        # Make pruning permanent (remove masks) to get actual sparse weights
        remove_pruning_masks(model)
        
        # Calculate real MACs by measuring actual operations with sparse weights
        print("Calculating real MACs with sparse weights...")
        macs = calculate_real_macs(model, example_inputs)
        
        print(model)
        print("#Params: {:.4f} M => {:.4f} M".format(base_params/1e6, params/1e6))
        print("#MACS: {:.4f} G => {:.4f} G".format(base_macs/1e9, macs/1e9))
        print("Sparsity ratio: {:.2f}%".format((1 - params/base_params) * 100))

    pipeline.save_pretrained(args.save_path)
    if args.pruning_ratio > 0:
        os.makedirs(os.path.join(args.save_path, "pruned"), exist_ok=True)
        torch.save(model, os.path.join(args.save_path, "pruned", "unet_weight_pruned.pth"))

    # Sampling images from the pruned model
    pipeline = DDIMPipeline(
        unet = model,
        scheduler = DDIMScheduler.from_pretrained(args.save_path, subfolder="scheduler")
    )
    with torch.no_grad():
        generator = torch.Generator(device=pipeline.device).manual_seed(0)
        pipeline.to("cuda")
        images = pipeline(num_inference_steps=100, batch_size=args.batch_size, generator=generator, output_type="numpy").images
        os.makedirs(os.path.join(args.save_path, 'vis'), exist_ok=True)
        torchvision.utils.save_image(torch.from_numpy(images).permute([0, 3, 1, 2]), "{}/vis/after_weight_pruning.png".format(args.save_path))
