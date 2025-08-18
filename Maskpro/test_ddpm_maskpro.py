import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from diffusers import DDPMPipeline, DDIMPipeline, DDIMScheduler
import torchvision
from PIL import Image
import sys
sys.path.append('..')
import utils

# Import wrapper for mask functionality
from wrapper_diffusion import mask_wrapper_diffusion

# Import FID calculation
try:
    from scipy import linalg
    from torchvision.models import inception_v3
    import torch.nn.functional as F
    FID_AVAILABLE = True
except ImportError:
    print("Warning: FID calculation requires scipy and torchvision. Install with: pip install scipy")
    FID_AVAILABLE = False

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to trained MaskPro checkpoint (can be directory or checkpoint subdirectory)')
parser.add_argument('--original_model', type=str, default='../pretrained/ddpm_ema_cifar10',
                    help='Path to ORIGINAL complete DDPM model (NOT pruned) for proper baseline')
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate')
parser.add_argument('--num_inference_steps', type=int, default=100, help='Number of inference steps')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size for generation')
parser.add_argument('--device', type=str, default='cuda:0', help='Device')
parser.add_argument('--output_dir', type=str, default='test_results', help='Output directory')
parser.add_argument('--compute_fid', action='store_true', help='Compute FID score')

args = parser.parse_args()

def generate_samples(pipeline, num_samples, batch_size, num_inference_steps, device, output_dir, model_name):
    """Generate samples from the model"""
    os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)
    
    # Ensure pipeline is on correct device
    pipeline = pipeline.to(device)
    
    all_images = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Generating {num_samples} samples with {model_name}...")
    
    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            generator = torch.Generator(device=device).manual_seed(i)
            
            images = pipeline(
                batch_size=current_batch_size,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="numpy"
            ).images
            
            # Save individual images
            for j, img in enumerate(images):
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                img_pil.save(os.path.join(output_dir, model_name, f"sample_{i*batch_size + j:04d}.png"))
            
            all_images.append(images)
            
            if (i + 1) % 5 == 0:
                print(f"Generated {(i+1) * batch_size} / {num_samples} samples")
    
    # Concatenate all images
    all_images = np.concatenate(all_images, axis=0)[:num_samples]
    
    # Create a grid of samples
    grid_images = torch.from_numpy(all_images).permute(0, 3, 1, 2)
    grid = torchvision.utils.make_grid(grid_images[:64], nrow=8, normalize=True, scale_each=True)
    torchvision.utils.save_image(grid, os.path.join(output_dir, f"{model_name}_grid.png"))
    
    print(f"Saved {len(all_images)} samples to {os.path.join(output_dir, model_name)}")
    return all_images

def get_inception_model(device):
    """Load pretrained Inception-v3 model for FID calculation"""
    # Use the new weights parameter instead of deprecated pretrained
    from torchvision.models import Inception_V3_Weights
    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    inception.fc = torch.nn.Identity()  # Remove final classification layer
    inception.eval()
    inception = inception.to(device)
    return inception

def get_inception_features(images, inception_model, device, batch_size=50):
    """Extract Inception features from images"""
    inception_model.eval()
    features = []
    
    # Convert images to tensor if needed
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).permute(0, 3, 1, 2)  # NHWC -> NCHW
    
    # Move images to device
    images = images.to(device)
    
    # Normalize to [0, 1] if needed
    if images.max() > 1:
        images = images.float() / 255.0
    
    # Ensure float type
    images = images.float()
    
    # Resize to 299x299 for Inception
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    
    # Normalize for ImageNet pretrained model (move tensors to device)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    images = (images - mean) / std
    
    num_batches = (len(images) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(images))
            batch = images[start_idx:end_idx]
            
            batch_features = inception_model(batch)
            features.append(batch_features.cpu().numpy())
    
    return np.concatenate(features, axis=0)

def calculate_fid(real_features, fake_features):
    """Calculate FID score between real and fake features"""
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def compute_fid_score(real_images, fake_images, device):
    """Compute FID score between real and fake images"""
    if not FID_AVAILABLE:
        print("FID calculation not available. Please install required packages.")
        return None
    
    print("Computing FID score...")
    
    # Load Inception model
    inception_model = get_inception_model(device)
    
    # Extract features
    print("Extracting features from real images...")
    real_features = get_inception_features(real_images, inception_model, device)
    
    print("Extracting features from generated images...")
    fake_features = get_inception_features(fake_images, inception_model, device)
    
    # Calculate FID
    fid_score = calculate_fid(real_features, fake_features)
    
    print(f"FID Score: {fid_score:.4f}")
    return fid_score

def compute_model_stats(model):
    """Compute model statistics"""
    total_params = 0
    zero_params = 0
    
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            weight = module.weight.data
            total_params += weight.numel()
            zero_params += (weight == 0).sum().item()
    
    sparsity = zero_params / total_params if total_params > 0 else 0.0
    return total_params, zero_params, sparsity

def evaluate_model_loss(pipeline, scheduler, dataset, device, num_batches=50):
    """Evaluate model loss on test data"""
    model = pipeline.unet
    model.eval()
    
    # Ensure model is on correct device
    model = model.to(device)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    total_loss = 0
    count = 0
    
    print("Evaluating model loss...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            if isinstance(batch, (list, tuple)):
                clean_images = batch[0]
            else:
                clean_images = batch
                
            clean_images = clean_images.to(device)
            
            # Ensure proper image format
            if clean_images.min() >= 0 and clean_images.max() <= 1:
                clean_images = clean_images * 2 - 1
            
            batch_size = clean_images.shape[0]
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device)
            noise = torch.randn_like(clean_images)
            noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
            
            noise_pred = model(noisy_images, timesteps).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            total_loss += loss.item()
            count += 1
    
    avg_loss = total_loss / count if count > 0 else 0
    print(f"Average reconstruction loss: {avg_loss:.6f}")
    return avg_loss

if __name__ == '__main__':
    print("CORRECTED Testing DDPM MaskPro model...")
    print("=== CORRECTED METHODOLOGY ===")
    print("Comparison: ORIGINAL_MODEL + LEARNED_MASK vs ORIGINAL_MODEL + INITIAL_MASK")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Original model: {args.original_model}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 1. Load ORIGINAL complete model for baseline (with initial masks)
    print("\n1. Loading ORIGINAL complete model for baseline...")
    try:
        baseline_pipeline = DDPMPipeline.from_pretrained(args.original_model)
        baseline_model = baseline_pipeline.unet.to(device)
        baseline_scheduler = baseline_pipeline.scheduler
        
        # Load and apply INITIAL masks to create proper baseline
        initial_mask_path = "initial_mask_diffusion"
        learned_mask_path = "learned_mask_diffusion"
        
        if os.path.exists(initial_mask_path):
            initial_mask_name_list = [f.replace(".pt", "") for f in os.listdir(initial_mask_path) if f.endswith('.pt')]
            print(f"Found {len(initial_mask_name_list)} initial masks")
            
            # Apply initial masks to create baseline: ORIGINAL_MODEL + INITIAL_MASK
            mask_wrapper_diffusion(baseline_model, initial_mask_name_list, [], 
                                 logits_magnitude=10.0, targets=['down_blocks', 'up_blocks', 'mid_block'])
            print("✓ Baseline model: ORIGINAL_MODEL + INITIAL_MASK")
        else:
            print("Error: No initial masks found!")
            exit(1)
        
        # Convert to DDIM for faster sampling
        ddim_scheduler = DDIMScheduler.from_config(baseline_scheduler.config)
        baseline_pipeline = DDIMPipeline(unet=baseline_model, scheduler=ddim_scheduler)
        
        print("Baseline model loaded successfully")
    except Exception as e:
        print(f"Error loading baseline model: {e}")
        exit(1)
    
    # 2. Load trained model (ORIGINAL_MODEL + LEARNED_MASK)
    print("\n2. Loading trained MaskPro model...")
    
    # Auto-detect checkpoint path
    checkpoint_dir = args.checkpoint_path
    if not os.path.isfile(os.path.join(checkpoint_dir, "model_index.json")):
        # Try checkpoint subdirectory
        checkpoint_subdir = os.path.join(checkpoint_dir, "checkpoint")
        if os.path.isfile(os.path.join(checkpoint_subdir, "model_index.json")):
            checkpoint_dir = checkpoint_subdir
            print(f"Found model in checkpoint subdirectory: {checkpoint_dir}")
        else:
            print(f"Error: No model_index.json found in {checkpoint_dir} or {checkpoint_subdir}")
            exit(1)
    
    try:
        trained_pipeline = DDPMPipeline.from_pretrained(checkpoint_dir)
        trained_scheduler = trained_pipeline.scheduler
        
        # Move pipeline to device
        trained_pipeline = trained_pipeline.to(device)
        
        # Convert to DDIM for faster sampling
        ddim_scheduler = DDIMScheduler.from_config(trained_scheduler.config)
        trained_pipeline = DDIMPipeline(unet=trained_pipeline.unet, scheduler=ddim_scheduler)
        
        print("✓ Trained model: ORIGINAL_MODEL + LEARNED_MASK")
        print("Trained model loaded successfully")
    except Exception as e:
        print(f"Error loading trained model: {e}")
        exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Compute model statistics
    print("\n=== Model Statistics ===")
    trained_total, trained_zero, trained_sparsity = compute_model_stats(trained_pipeline.unet)
    baseline_total, baseline_zero, baseline_sparsity = compute_model_stats(baseline_pipeline.unet)
    
    print(f"Baseline model (ORIGINAL+INITIAL_MASK) - Total: {baseline_total:,}, Zero: {baseline_zero:,}, Sparsity: {baseline_sparsity:.4f}")
    print(f"Trained model (ORIGINAL+LEARNED_MASK) - Total: {trained_total:,}, Zero: {trained_zero:,}, Sparsity: {trained_sparsity:.4f}")
    
    # Load dataset for evaluation
    dataset = utils.get_dataset(args.dataset)
    
    # Evaluate reconstruction loss - CORRECTED COMPARISON
    print("\n=== CORRECTED Loss Evaluation ===")
    print("Comparing: ORIGINAL_MODEL + LEARNED_MASK vs ORIGINAL_MODEL + INITIAL_MASK")
    
    baseline_loss = evaluate_model_loss(baseline_pipeline, trained_scheduler, dataset, device)
    trained_loss = evaluate_model_loss(trained_pipeline, trained_scheduler, dataset, device)
    
    improvement = baseline_loss - trained_loss
    relative_improvement = improvement / baseline_loss * 100 if baseline_loss != 0 else 0
    
    print(f"\nCORRECTED Results:")
    print(f"Baseline loss (ORIGINAL+INITIAL_MASK): {baseline_loss:.6f}")
    print(f"Trained loss (ORIGINAL+LEARNED_MASK): {trained_loss:.6f}")
    print(f"Loss improvement: {improvement:.6f} ({relative_improvement:.2f}%)")
    
    if improvement > 0:
        print("✓ POSITIVE IMPROVEMENT: MaskPro training improved the model!")
    else:
        print("✗ NEGATIVE IMPROVEMENT: Initial masks were already quite good")
        print("  This suggests magnitude pruning produced high-quality masks")
        print("  that are difficult to improve with 2:4 structured constraints")
    
    # Generate samples
    print("\n=== Sample Generation ===")
    baseline_samples = generate_samples(baseline_pipeline, args.num_samples, args.batch_size, 
                                      args.num_inference_steps, device, args.output_dir, "baseline_initial_mask")
    
    trained_samples = generate_samples(trained_pipeline, args.num_samples, args.batch_size, 
                                     args.num_inference_steps, device, args.output_dir, "trained_learned_mask")
    
    # Compute FID scores if requested
    fid_results = {}
    if args.compute_fid and FID_AVAILABLE:
        print("\n=== CORRECTED FID Score Computation ===")
        
        # Load real CIFAR-10 images for reference
        print("Loading real CIFAR-10 images...")
        real_dataset = utils.get_dataset(args.dataset)
        
        real_images = []
        real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=True)
        
        for batch_idx, batch in enumerate(real_loader):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            # Convert to numpy format [0, 1]
            images = images.numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]
            images = np.clip(images, 0, 1)
            
            real_images.append(images)
            
            if len(real_images) * args.batch_size >= args.num_samples:
                break
        
        real_images = np.concatenate(real_images, axis=0)[:args.num_samples]
        
        # Compute FID for baseline (ORIGINAL+INITIAL_MASK)
        baseline_fid = compute_fid_score(real_images, baseline_samples, device)
        if baseline_fid is not None:
            fid_results['baseline_fid'] = baseline_fid
        
        # Compute FID for trained (ORIGINAL+LEARNED_MASK)
        trained_fid = compute_fid_score(real_images, trained_samples, device)
        if trained_fid is not None:
            fid_results['trained_fid'] = trained_fid
        
        # Compute FID improvement
        if baseline_fid is not None and trained_fid is not None:
            fid_improvement = baseline_fid - trained_fid
            fid_results['fid_improvement'] = fid_improvement
            print(f"\nCORRECTED FID Comparison:")
            print(f"Baseline FID (ORIGINAL+INITIAL_MASK): {baseline_fid:.4f}")
            print(f"Trained FID (ORIGINAL+LEARNED_MASK): {trained_fid:.4f}")
            print(f"FID improvement: {fid_improvement:.4f}")
            
            if fid_improvement > 0:
                print("✓ POSITIVE FID IMPROVEMENT: Better image quality!")
            else:
                print("✗ NEGATIVE FID IMPROVEMENT: Initial masks produced better images")
    
    elif args.compute_fid and not FID_AVAILABLE:
        print("\n=== FID Score Computation ===")
        print("FID calculation not available. Please install scipy with: pip install scipy")
    
    # Save CORRECTED evaluation results
    results = {
        'methodology': 'CORRECTED: ORIGINAL_MODEL + LEARNED_MASK vs ORIGINAL_MODEL + INITIAL_MASK',
        'checkpoint_path': args.checkpoint_path,
        'original_model': args.original_model,
        'baseline_stats': {
            'description': 'ORIGINAL_MODEL + INITIAL_MASK',
            'total_params': baseline_total,
            'zero_params': baseline_zero,
            'sparsity': baseline_sparsity,
            'loss': baseline_loss
        },
        'trained_stats': {
            'description': 'ORIGINAL_MODEL + LEARNED_MASK',
            'total_params': trained_total,
            'zero_params': trained_zero,
            'sparsity': trained_sparsity,
            'loss': trained_loss
        },
        'corrected_results': {
            'loss_improvement': improvement,
            'relative_improvement_percent': relative_improvement,
            'improvement_positive': improvement > 0
        },
        'num_samples_generated': len(trained_samples),
        'num_inference_steps': args.num_inference_steps
    }
    
    # Add FID results
    if fid_results:
        results['fid_results'] = fid_results
    
    import json
    with open(os.path.join(args.output_dir, "corrected_evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== CORRECTED Evaluation Complete ===")
    print(f"Results saved to {args.output_dir}")
    print(f"Generated samples: {args.num_samples}")
    print(f"CORRECTED Loss improvement: {improvement:.6f} ({relative_improvement:.2f}%)")
    
    if fid_results and 'fid_improvement' in fid_results:
        print(f"CORRECTED FID improvement: {fid_results['fid_improvement']:.4f}")
    
    # Summary
    print(f"\n=== CORRECTED SUMMARY ===")
    print(f"Methodology: ORIGINAL_MODEL + LEARNED_MASK vs ORIGINAL_MODEL + INITIAL_MASK")
    print(f"Loss improvement: {improvement:.6f} ({'POSITIVE' if improvement > 0 else 'NEGATIVE'})")
    if 'fid_improvement' in fid_results:
        fid_imp = fid_results['fid_improvement']
        print(f"FID improvement: {fid_imp:.4f} ({'POSITIVE' if fid_imp > 0 else 'NEGATIVE'})")
    
    if improvement < 0:
        print("\n=== Analysis ===")
        print("Negative improvement suggests that:")
        print("1. Initial masks (from magnitude pruning) were already quite good")
        print("2. 2:4 structured constraint is more restrictive than magnitude pruning")
        print("3. The learning rate or training strategy may need adjustment")
        print("4. Consider starting from random or uniform masks instead")