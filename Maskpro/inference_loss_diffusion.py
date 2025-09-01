import os
import time
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from diffusers import DDPMPipeline, DDPMScheduler
from tqdm import tqdm
import sys
sys.path.append('..')
import utils

from wrapper_diffusion import mask_wrapper_diffusion

parser = argparse.ArgumentParser()
parser.add_argument('--original_model', type=str, default='../pretrained/ddpm_ema_cifar10',
                    help='Path to ORIGINAL complete DDPM model')
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
parser.add_argument('--dataset_size', type=int, default=512, help='Number of samples for loss computation')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
parser.add_argument('--max_batches', type=int, default=50, help='Maximum number of batches to process')
parser.add_argument('--device', type=str, default='cuda:0', help='Device')
parser.add_argument('--targets', nargs='+', type=str, 
                    default=['down_blocks', 'up_blocks', 'mid_block'], 
                    help='Target layer prefixes for training')
args = parser.parse_args()

def compute_diffusion_loss(model, scheduler, clean_images, device):
    """Compute the diffusion loss (noise prediction loss)"""
    batch_size = clean_images.shape[0]
    
    # Sample random timesteps
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device)
    
    # Sample noise
    noise = torch.randn_like(clean_images, device=device)
    
    # Add noise to images according to timestep
    noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
    
    # Predict noise
    with torch.no_grad():
        noise_pred = model(noisy_images, timesteps).sample
    
    # Compute MSE loss
    loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='mean')
    return loss.item()

if __name__ == '__main__':
    print(f"Computing baseline losses...")
    print(f"Original model: {args.original_model}")
    print(f"Mask files from: initial_mask_diffusion/")
    
    # Load ORIGINAL complete model (NOT the pruned model!)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    try:
        if os.path.isfile(os.path.join(args.original_model, "model_index.json")):
            pipeline = DDPMPipeline.from_pretrained(args.original_model)
            model = pipeline.unet.to(device)
            scheduler = pipeline.scheduler
            print(f"✓ Original complete model loaded from {args.original_model}")
        else:
            raise FileNotFoundError(f"Could not find original model at {args.original_model}")
            
        print(f"Model loaded on {device}")
        
    except Exception as e:
        print(f"Error loading original model: {e}")
        print(f"Make sure the original DDPM model exists at: {args.original_model}")
        print(f"Available models in ../pretrained/:")
        pretrained_dir = "../pretrained"
        if os.path.exists(pretrained_dir):
            for model_name in os.listdir(pretrained_dir):
                if os.path.isdir(os.path.join(pretrained_dir, model_name)):
                    print(f"  - {model_name}")
        exit(1)
    
    model.eval()
    
    # Load initial masks and wrap model if masks exist
    initial_mask_path = "initial_mask_diffusion"
    learned_mask_path = "learned_mask_diffusion"
    
    initial_mask_name_list = []
    learned_mask_name_list = []
    
    if os.path.exists(initial_mask_path):
        initial_mask_name_list = [f.replace(".pt", "") for f in os.listdir(initial_mask_path) if f.endswith('.pt')]
        print(f"Found {len(initial_mask_name_list)} initial masks")
    
    if os.path.exists(learned_mask_path):
        learned_mask_name_list = [f.replace(".pt", "") for f in os.listdir(learned_mask_path) if f.endswith('.pt')]
        print(f"Found {len(learned_mask_name_list)} learned masks")
    
    # Wrap model with masks if available
    if initial_mask_name_list:
        print(f"Applying initial masks to ORIGINAL model...")
        print(f"This computes baseline: ORIGINAL_MODEL + INITIAL_MASK")
        mask_wrapper_diffusion(model, initial_mask_name_list, learned_mask_name_list, 
                             logits_magnitude=10.0, targets=args.targets)
    else:
        print("Warning: No initial masks found!")
        print("This will compute loss for original unmasked model")
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset.lower() == 'cifar10':
        resolution = 32
    else:
        resolution = 64
        
    dataset = utils.get_dataset(args.dataset)
    
    # Limit dataset size
    if len(dataset) > args.dataset_size:
        indices = torch.randperm(len(dataset))[:args.dataset_size]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Dataset size: {len(dataset)}, Batches: {len(dataloader)}")
    
    # Compute losses
    model.eval()
    loss_list = []
    
    print(f"Computing baseline losses...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing losses")):
            if batch_idx >= args.max_batches:
                break
                
            # Handle different dataset formats
            if isinstance(batch, (list, tuple)):
                clean_images = batch[0]
            else:
                clean_images = batch
                
            clean_images = clean_images.to(device)
            
            # Ensure proper image format (should be in [-1, 1] range)
            if clean_images.min() >= 0 and clean_images.max() <= 1:
                clean_images = clean_images * 2 - 1  # Convert [0,1] to [-1,1]
            
            loss = compute_diffusion_loss(model, scheduler, clean_images, device)
            loss_list.append(loss)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Loss = {loss:.6f}")
    
    # Save results
    loss_array = np.array(loss_list)
    
    # Create baseline_losses directory if it doesn't exist
    baseline_dir = "baseline_losses"
    os.makedirs(baseline_dir, exist_ok=True)
    
    # Create filename based on parameters
    filename = f"inference_loss_diffusion_{args.dataset}_bs{args.batch_size}_size{len(loss_list)}.npy"
    filepath = os.path.join(baseline_dir, filename)
    np.save(filepath, loss_array)
    
    # Print statistics
    print(f"\n=== CORRECTED Baseline Loss Summary ===")
    print(f"Dataset: {args.dataset}")
    print(f"Original model: {args.original_model}")
    print(f"Mask source: initial_mask_diffusion/ (extracted from pruned model)")
    print(f"Computation: ORIGINAL_MODEL + EXTRACTED_INITIAL_MASK")
    print(f"Batches processed: {len(loss_list)}")
    print(f"Mean baseline loss: {loss_array.mean():.6f}")
    print(f"Std baseline loss: {loss_array.std():.6f}")
    print(f"Min baseline loss: {loss_array.min():.6f}")
    print(f"Max baseline loss: {loss_array.max():.6f}")
    print(f"Results saved to: {filepath}")
    
    # Save summary
    summary_file = os.path.join(baseline_dir, filename.replace('.npy', '_summary.txt'))
    with open(summary_file, 'w') as f:
        f.write(f"CORRECTED Baseline Loss Computation Summary\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Original model: {args.original_model}\n")
        f.write(f"Mask source: initial_mask_diffusion/ (extracted from pruned model)\n")
        f.write(f"Computation: ORIGINAL_MODEL + EXTRACTED_INITIAL_MASK\n")
        f.write(f"Batches processed: {len(loss_list)}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Mean baseline loss: {loss_array.mean():.6f}\n")
        f.write(f"Std baseline loss: {loss_array.std():.6f}\n")
        f.write(f"Min baseline loss: {loss_array.min():.6f}\n")
        f.write(f"Max baseline loss: {loss_array.max():.6f}\n")
        f.write(f"Note: This baseline uses ORIGINAL complete model + extracted mask\n")
        f.write(f"This matches the original MaskPro methodology correctly\n")
        import datetime
        f.write(f"Computation date: {datetime.datetime.now()}\n")
    
    print(f"Summary saved to: {summary_file}")