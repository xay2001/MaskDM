from diffusers import LDMPipeline, DDIMPipeline, DDIMScheduler
from diffusers.models import UNet2DModel
from diffusers import VQModel
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
parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained LDM model")
parser.add_argument("--save_path", type=str, required=True, help="Path to save pruned model")
parser.add_argument("--pruning_ratio", type=float, default=0.3, help="Pruning ratio for weights")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for test generation")
parser.add_argument("--device", type=str, default='cuda', help="Device to use")
parser.add_argument("--pruner", type=str, default='magnitude', choices=['magnitude', 'random'], 
                    help="Pruning method: magnitude or random")

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

def get_prunable_modules(model, model_name="unet"):
    """Get all prunable modules and categorize them by importance
    
    Protection strategy (similar to ddpm_weight_prune.py):
    - conv_in: input convolution layer
    - conv_out: output convolution layer
    - time_emb/time_embedding: time step embedding layers
    - class_emb: class embedding layers (if exists)
    """
    prunable_modules = []
    protected_modules = []
    
    # Collect all named modules
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Protect critical layers - following ddpm_weight_prune.py strategy
            if ('conv_in' in name or 
                'conv_out' in name or 
                'time_emb' in name or
                'time_embedding' in name or
                'class_emb' in name):
                protected_modules.append((name, module))
                print(f"[{model_name}] Protecting layer: {name} (shape: {module.weight.shape})")
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

def print_layer_sparsity(model, model_name="Model"):
    """Print sparsity information for each layer"""
    print(f"\n=== {model_name} Layer-wise Sparsity ===")
    total_params = 0
    total_zero = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            sparsity = (weight == 0).float().mean().item()
            num_params = weight.numel()
            num_zero = (weight == 0).sum().item()
            
            total_params += num_params
            total_zero += num_zero
            
            if sparsity > 0:  # Only print layers with non-zero sparsity
                print(f"  {name}: {sparsity:.4f} ({weight.shape})")
    
    overall_sparsity = total_zero / total_params if total_params > 0 else 0.0
    print(f"  Overall: {overall_sparsity:.4f} ({total_zero}/{total_params})")

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__=='__main__':
    
    # Loading pretrained model
    print("="*80)
    print(f"Loading pretrained LDM model from {args.model_path}")
    print("="*80)
    
    # Load all components
    unet = UNet2DModel.from_pretrained(args.model_path, subfolder="unet")
    vqvae = VQModel.from_pretrained(args.model_path, subfolder="vqvae")
    scheduler = DDIMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    
    # Set device
    torch_device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {torch_device}")
    
    unet = unet.to(torch_device)
    vqvae = vqvae.to(torch_device)
    
    unet.eval()
    vqvae.eval()
    
    # Calculate initial parameters
    print("\n" + "="*80)
    print("Initial Model Statistics")
    print("="*80)
    
    unet_total, unet_trainable = count_parameters(unet)
    vqvae_total, vqvae_trainable = count_parameters(vqvae)
    total_params = unet_total + vqvae_total
    
    print(f"UNet parameters: {unet_total:,} total, {unet_trainable:,} trainable")
    print(f"VQVAE parameters: {vqvae_total:,} total, {vqvae_trainable:,} trainable")
    print(f"Total LDM parameters: {total_params:,}")
    
    # Apply weight pruning to UNet only (VQVAE is fully preserved)
    if args.pruning_ratio > 0:
        print("\n" + "="*80)
        print(f"Applying {args.pruner} weight pruning to UNet")
        print(f"Pruning ratio: {args.pruning_ratio}")
        print(f"VQVAE: FULLY PRESERVED (no pruning)")
        print("="*80)
        
        # Get prunable and protected modules for UNet
        prunable_modules, protected_modules = get_prunable_modules(unet, "UNet")
        
        print(f"\nUNet statistics:")
        print(f"  Prunable layers: {len(prunable_modules)}")
        print(f"  Protected layers: {len(protected_modules)}")
        
        # Apply pruning to UNet
        total_sparsity = 0
        pruned_layers = 0
        
        print("\nPruning UNet layers...")
        for name, module in tqdm(prunable_modules, desc="Pruning UNet"):
            if args.pruner == 'magnitude':
                sparsity = apply_magnitude_pruning(module, args.pruning_ratio)
            elif args.pruner == 'random':
                sparsity = apply_random_pruning(module, args.pruning_ratio)
            
            if sparsity > 0:
                total_sparsity += sparsity
                pruned_layers += 1
        
        # Calculate final statistics
        unet_sparsity = calculate_model_sparsity(unet)
        vqvae_sparsity = calculate_model_sparsity(vqvae)  # Should be 0
        avg_layer_sparsity = total_sparsity / pruned_layers if pruned_layers > 0 else 0
        
        print("\n" + "="*80)
        print("Pruning Results")
        print("="*80)
        print(f"Pruning method: {args.pruner}")
        print(f"Target pruning ratio: {args.pruning_ratio}")
        print(f"\nUNet:")
        print(f"  Layers pruned: {pruned_layers}")
        print(f"  Average layer sparsity: {avg_layer_sparsity:.4f}")
        print(f"  Overall UNet sparsity: {unet_sparsity:.4f}")
        print(f"\nVQVAE:")
        print(f"  Sparsity: {vqvae_sparsity:.4f} (preserved)")
        
        # Calculate effective sparsity for entire LDM
        total_model_sparsity = (unet_sparsity * unet_total + vqvae_sparsity * vqvae_total) / total_params
        print(f"\nOverall LDM sparsity: {total_model_sparsity:.4f}")
        
        # Print detailed layer information
        print_layer_sparsity(unet, "UNet")
        
    # Save the pruned model
    print("\n" + "="*80)
    print(f"Saving pruned LDM model to {args.save_path}")
    print("="*80)
    
    # Create LDM pipeline
    pipeline = LDMPipeline(
        unet=unet,
        vqvae=vqvae,
        scheduler=scheduler,
    ).to(torch_device)
    
    pipeline.save_pretrained(args.save_path)
    
    if args.pruning_ratio > 0:
        # Save pruned UNet separately
        os.makedirs(os.path.join(args.save_path, "pruned"), exist_ok=True)
        torch.save(unet, os.path.join(args.save_path, "pruned", "unet_pruned.pth"))
        
        # Save pruning statistics
        stats = {
            'pruning_method': args.pruner,
            'target_ratio': args.pruning_ratio,
            'unet_layers_pruned': pruned_layers,
            'unet_avg_layer_sparsity': avg_layer_sparsity,
            'unet_overall_sparsity': unet_sparsity,
            'vqvae_sparsity': vqvae_sparsity,
            'ldm_overall_sparsity': total_model_sparsity,
            'unet_initial_params': unet_total,
            'vqvae_initial_params': vqvae_total,
            'total_initial_params': total_params,
            'unet_prunable_layers': len(prunable_modules),
            'unet_protected_layers': len(protected_modules)
        }
        
        import json
        with open(os.path.join(args.save_path, "pruning_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved pruning statistics to {os.path.join(args.save_path, 'pruning_stats.json')}")
    
    # Test the pruned model with sample generation
    print("\n" + "="*80)
    print("Testing pruned LDM with sample generation")
    print("="*80)
    
    try:
        with torch.no_grad():
            generator = torch.Generator(device=torch_device).manual_seed(0)
            
            print(f"Generating {args.batch_size} images with 100 steps...")
            images = pipeline(
                num_inference_steps=100,
                batch_size=args.batch_size,
                generator=generator,
                output_type="numpy"
            ).images
            
            # Save generated images
            os.makedirs(os.path.join(args.save_path, 'vis'), exist_ok=True)
            save_path = os.path.join(args.save_path, 'vis', 'after_weight_pruning.png')
            
            torchvision.utils.save_image(
                torch.from_numpy(images).permute([0, 3, 1, 2]),
                save_path
            )
            
            print(f"✓ Sample generation successful!")
            print(f"✓ Saved generated images to {save_path}")
            
    except Exception as e:
        print(f"✗ Sample generation failed: {e}")
        print("This might indicate that the pruning ratio is too aggressive.")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("LDM Weight Pruning Completed!")
    print("="*80)
    print(f"Pruned model saved to: {args.save_path}")
    if args.pruning_ratio > 0:
        print(f"UNet sparsity: {unet_sparsity:.2%}")
        print(f"Overall LDM sparsity: {total_model_sparsity:.2%}")

