from diffusers import DiffusionPipeline, DDPMPipeline, DDIMPipeline, DDIMScheduler, DDPMScheduler
from diffusers.models import UNet2DModel
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
import os
from glob import glob
from PIL import Image
import accelerate
import utils
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None, help="path to an image folder")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--pruning_ratio", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--pruner", type=str, default='magnitude', choices=['magnitude', 'random'])

args = parser.parse_args()

def apply_magnitude_pruning(module, pruning_ratio):
    """Apply magnitude-based weight pruning to a module"""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        weight = module.weight.data
        # Flatten the weight tensor
        weight_flat = weight.view(-1)
        # Calculate threshold for magnitude pruning
        num_weights = weight_flat.numel()
        num_prune = int(num_weights * pruning_ratio)
        
        if num_prune > 0:
            # Get the threshold value (smallest magnitude to keep)
            weight_abs = torch.abs(weight_flat)
            threshold = torch.topk(weight_abs, num_weights - num_prune, largest=True)[0][-1]
            
            # Create mask: keep weights with magnitude >= threshold
            mask = weight_abs >= threshold
            
            # Apply mask to weights
            weight_flat[~mask] = 0
            module.weight.data = weight_flat.view(weight.shape)
            
            actual_sparsity = (weight_flat == 0).float().mean().item()
            return actual_sparsity
    return 0.0

def apply_random_pruning(module, pruning_ratio):
    """Apply random weight pruning to a module"""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        weight = module.weight.data
        # Flatten the weight tensor
        weight_flat = weight.view(-1)
        num_weights = weight_flat.numel()
        num_prune = int(num_weights * pruning_ratio)
        
        if num_prune > 0:
            # Randomly select indices to prune
            indices = torch.randperm(num_weights)[:num_prune]
            weight_flat[indices] = 0
            module.weight.data = weight_flat.view(weight.shape)
            
            actual_sparsity = (weight_flat == 0).float().mean().item()
            return actual_sparsity
    return 0.0

def get_prunable_modules(model):
    """Get all prunable modules and categorize them by importance"""
    prunable_modules = []
    protected_modules = []
    
    # Collect all named modules
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Protect critical layers
            if ('conv_in' in name or 
                'conv_out' in name or 
                'time_emb' in name or
                'class_emb' in name):
                protected_modules.append((name, module))
                print(f"Protecting layer: {name}")
            else:
                prunable_modules.append((name, module))
    
    return prunable_modules, protected_modules

def calculate_model_sparsity(model):
    """Calculate overall model sparsity"""
    total_params = 0
    zero_params = 0
    
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            total_params += weight.numel()
            zero_params += (weight == 0).sum().item()
    
    return zero_params / total_params if total_params > 0 else 0.0

def print_layer_sparsity(model):
    """Print sparsity information for each layer"""
    print("\n=== Layer-wise Sparsity ===")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            sparsity = (weight == 0).float().mean().item()
            print(f"{name}: {sparsity:.4f} ({weight.shape})")

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

    # Calculate initial parameters
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    initial_total, initial_trainable = count_parameters(model)
    print(f"Initial parameters: {initial_total:,} total, {initial_trainable:,} trainable")
    
    if args.pruning_ratio > 0:
        print(f"\nApplying {args.pruner} pruning with ratio {args.pruning_ratio}")
        
        # Get prunable and protected modules
        prunable_modules, protected_modules = get_prunable_modules(model)
        
        print(f"Found {len(prunable_modules)} prunable layers and {len(protected_modules)} protected layers")
        
        # Apply pruning
        total_sparsity = 0
        pruned_layers = 0
        
        for name, module in tqdm(prunable_modules, desc="Pruning layers"):
            if args.pruner == 'magnitude':
                sparsity = apply_magnitude_pruning(module, args.pruning_ratio)
            elif args.pruner == 'random':
                sparsity = apply_random_pruning(module, args.pruning_ratio)
            
            if sparsity > 0:
                total_sparsity += sparsity
                pruned_layers += 1
                print(f"Pruned {name}: {sparsity:.4f} sparsity")
        
        # Calculate final statistics
        overall_sparsity = calculate_model_sparsity(model)
        avg_layer_sparsity = total_sparsity / pruned_layers if pruned_layers > 0 else 0
        
        print(f"\n=== Pruning Results ===")
        print(f"Pruning method: {args.pruner}")
        print(f"Target pruning ratio: {args.pruning_ratio}")
        print(f"Layers pruned: {pruned_layers}")
        print(f"Average layer sparsity: {avg_layer_sparsity:.4f}")
        print(f"Overall model sparsity: {overall_sparsity:.4f}")
        
        # Print detailed layer information
        print_layer_sparsity(model)
        
    # Save the pruned model
    print(f"\nSaving model to {args.save_path}")
    pipeline.save_pretrained(args.save_path)
    
    if args.pruning_ratio > 0:
        os.makedirs(os.path.join(args.save_path, "pruned"), exist_ok=True)
        torch.save(model, os.path.join(args.save_path, "pruned", "unet_pruned.pth"))
        
        # Save pruning statistics
        stats = {
            'pruning_method': args.pruner,
            'target_ratio': args.pruning_ratio,
            'layers_pruned': pruned_layers,
            'avg_layer_sparsity': avg_layer_sparsity,
            'overall_sparsity': overall_sparsity,
            'initial_params': initial_total,
            'prunable_layers': len(prunable_modules),
            'protected_layers': len(protected_modules)
        }
        
        import json
        with open(os.path.join(args.save_path, "pruning_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
    
    # Test the pruned model with a sample generation
    print("\nTesting pruned model with sample generation...")
    try:
        pipeline = DDIMPipeline(
            unet=model,
            scheduler=DDIMScheduler.from_pretrained(args.save_path, subfolder="scheduler")
        )
        
        with torch.no_grad():
            generator = torch.Generator(device=pipeline.device).manual_seed(0)
            pipeline.to(args.device)
            # Generate a smaller batch for testing
            test_batch_size = min(4, args.batch_size)
            images = pipeline(
                num_inference_steps=50, 
                batch_size=test_batch_size, 
                generator=generator, 
                output_type="numpy"
            ).images
            
            os.makedirs(os.path.join(args.save_path, 'vis'), exist_ok=True)
            torchvision.utils.save_image(
                torch.from_numpy(images).permute([0, 3, 1, 2]), 
                "{}/vis/after_weight_pruning.png".format(args.save_path)
            )
            print("Sample generation successful!")
            
    except Exception as e:
        print(f"Sample generation failed: {e}")
        print("This might indicate that the pruning ratio is too aggressive.")
    
    print("Weight pruning completed!")