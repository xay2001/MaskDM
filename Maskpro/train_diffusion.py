import os
import time
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from diffusers import DDPMPipeline, DDPMScheduler
import sys
sys.path.append('..')
import utils

# SwanLab monitoring
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    print("SwanLab not available. Install with: pip install swanlab")
    SWANLAB_AVAILABLE = False

from wrapper_diffusion import mask_wrapper_diffusion, mask_unwrapper_diffusion, generate_mask

parser = argparse.ArgumentParser()
parser.add_argument('--original_model', type=str, default='../pretrained/ddpm_ema_cifar10',
                    help='Path to ORIGINAL complete DDPM model')
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
parser.add_argument('--lr', type=float, default=50, help='Learning rate for logits')
parser.add_argument('--epoch', type=int, default=625, help='Training epochs (actually steps/16)')
parser.add_argument('--logits', type=float, default=10.0, help='Initial logits magnitude')
parser.add_argument('--dataset_size', type=int, default=512, help='Dataset size')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
parser.add_argument('--max_step', type=int, default=10000, help='Max training steps')
parser.add_argument('--device', type=str, default='cuda:0', help='Device')
parser.add_argument('--targets', nargs='+', type=str, 
                    default=['down_blocks', 'up_blocks', 'mid_block'], 
                    help='Target layer prefixes for training')
parser.add_argument('--save', action='store_true', help='Save learned masks')
parser.add_argument('--output_dir', type=str, default='results_diffusion', help='Output directory')

# SwanLab arguments
parser.add_argument('--project_name', type=str, default='DDPM-MaskPro', help='SwanLab project name')
parser.add_argument('--experiment_name', type=str, default=None, help='SwanLab experiment name')
parser.add_argument('--disable_swanlab', action='store_true', help='Disable SwanLab logging')

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
    noise_pred = model(noisy_images, timesteps).sample
    
    # Compute MSE loss
    loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='mean')
    return loss

if __name__ == '__main__':
    print(f"Starting DDPM MaskPro training")
    print(f"Original model: {args.original_model}")
    print(f"Dataset: {args.dataset}")
    print(f"Learning rate: {args.lr}")
    print(f"Logits magnitude: {args.logits}")
    
    # Initialize SwanLab
    use_swanlab = SWANLAB_AVAILABLE and not args.disable_swanlab
    if use_swanlab:
        # Generate experiment name if not provided
        if args.experiment_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
            args.experiment_name = f"ddpm_maskpro_lr{args.lr}_logits{args.logits}_{timestamp}"
        
        # Initialize SwanLab run
        swanlab.init(
            project=args.project_name,
            name=args.experiment_name,
            config={
                "original_model": args.original_model,
                "dataset": args.dataset,
                "learning_rate": args.lr,
                "epoch": args.epoch,
                "logits_magnitude": args.logits,
                "dataset_size": args.dataset_size,
                "batch_size": args.batch_size,
                "max_step": args.max_step,
                "device": args.device,
                "targets": args.targets,
                "save_masks": args.save,
                "output_dir": args.output_dir
            },
            tags=["DDPM", "MaskPro", "Diffusion", "Pruning"]
        )
        print(f"SwanLab initialized: Project={args.project_name}, Experiment={args.experiment_name}")
    else:
        print("SwanLab monitoring disabled")
    
    # Set random seeds
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
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
        print(f"If not available, convert it from HuggingFace with:")
        print(f"  bash ../tools/convert_cifar10_ddpm_ema.sh")
        exit(1)
    
    # Load masks
    initial_mask_path = "initial_mask_diffusion"
    learned_mask_path = "learned_mask_diffusion"
    
    initial_mask_name_list = []
    learned_mask_name_list = []
    
    if os.path.exists(initial_mask_path):
        initial_mask_name_list = [f.replace(".pt", "") for f in os.listdir(initial_mask_path) if f.endswith('.pt')]
        print(f"Found {len(initial_mask_name_list)} initial masks")
    else:
        print("No initial masks found! Please run get_mask_diffusion.py first.")
        exit(1)
    
    if os.path.exists(learned_mask_path):
        learned_mask_name_list = [f.replace(".pt", "").replace("_", ".") for f in os.listdir(learned_mask_path) if f.endswith('.pt')]
        print(f"Found {len(learned_mask_name_list)} learned masks")
    
    # Check target layers (handle "all" case and name format conversion)
    target_found = False
    if "all" in args.targets:
        target_found = True
        print(f"✓ Target 'all' specified - will optimize all {len(initial_mask_name_list)} layers with masks")
    else:
        for target in args.targets:
            # Convert target name format (dots to underscores for matching)
            target_underscore = target.replace(".", "_")
            if any(target_underscore in name for name in initial_mask_name_list):
                target_found = True
                break
        
        if not target_found:
            print(f"Warning: No target layers found matching {args.targets}")
            print("Available layers (showing dot notation equivalents):")
            for name in initial_mask_name_list[:10]:  # Show first 10
                dot_name = name.replace("_", ".")
                print(f"  {dot_name}")
            if len(initial_mask_name_list) > 10:
                print(f"  ... and {len(initial_mask_name_list) - 10} more")
    
    # Wrap model with masks
    print("Applying mask wrapper to ORIGINAL model...")
    print(f"Training strategy: ORIGINAL_MODEL + DYNAMIC_MASK")
    print(f"Baseline computed from: ORIGINAL_MODEL + INITIAL_MASK")
    mask_wrapper_diffusion(model, initial_mask_name_list, learned_mask_name_list, 
                         args.logits, args.targets)
    
    # Debug: Check if logits were added
    logits_count_after_wrapper = 0
    for module in model.modules():
        if hasattr(module, "logits"):
            logits_count_after_wrapper += 1
    print(f"Found {logits_count_after_wrapper} modules with logits after wrapper")
    
    # Load baseline losses
    print("Loading baseline losses...")
    loss_file = f"inference_loss_diffusion_{args.dataset}_bs{args.batch_size}_size*.npy"
    import glob
    loss_files = glob.glob(loss_file)
    if not loss_files:
        print(f"No baseline loss file found matching {loss_file}")
        print("Please run inference_loss_diffusion.py first.")
        exit(1)
    
    vanilla_loss_list = np.load(loss_files[0])
    print(f"Loaded {len(vanilla_loss_list)} baseline losses from {loss_files[0]}")
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = utils.get_dataset(args.dataset)
    
    # Limit dataset size and ensure compatibility
    if len(dataset) > args.dataset_size:
        indices = torch.randperm(len(dataset))[:args.dataset_size]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Ensure we don't exceed available baseline losses
    max_batches = len(vanilla_loss_list)
    effective_dataset_size = min(len(dataset), max_batches * args.batch_size)
    if len(dataset) > effective_dataset_size:
        dataset = torch.utils.data.Subset(dataset, range(effective_dataset_size))
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    print(f"Effective dataset size: {len(dataset)}, Batches: {len(dataloader)}")
    
    # Training setup
    model.eval()  # Keep in eval mode for consistency with baseline computation
    total_loss = 0
    delta = 0
    
    # Track times
    time_forward = 0
    time_backward = 0
    time_generate_mask = 0
    
    # Results tracking
    loss_list = []
    train_loss_list = []
    
    num_epochs = args.epoch
    max_step = args.max_step if num_epochs != 0 else 0
    
    # SwanLab: Log initial model statistics
    if use_swanlab:
        # Count trainable layers with logits
        trainable_layers = 0
        total_logits_params = 0
        for module in model.modules():
            if hasattr(module, "logits"):
                trainable_layers += 1
                total_logits_params += module.logits.numel()
        
        swanlab.log({
            "model/trainable_layers": trainable_layers,
            "model/total_logits_params": total_logits_params,
            "model/baseline_losses_count": len(vanilla_loss_list),
            "model/target_masks": len(initial_mask_name_list)
        })
    
    print(f"Starting training for {num_epochs} epochs...")
    
    global_step = 0
    start_time = time.time()  # Record start time for duration calculation
    
    for epoch in range(num_epochs):
        for step, (batch, vanilla_loss) in enumerate(zip(dataloader, vanilla_loss_list)):
            
            # Handle different dataset formats
            if isinstance(batch, (list, tuple)):
                clean_images = batch[0]
            else:
                clean_images = batch
                
            clean_images = clean_images.to(device)
            
            # Ensure proper image format (should be in [-1, 1] range)
            if clean_images.min() >= 0 and clean_images.max() <= 1:
                clean_images = clean_images * 2 - 1  # Convert [0,1] to [-1,1]
            
            # Forward pass
            with torch.no_grad():
                start = time.time()
                loss = compute_diffusion_loss(model, scheduler, clean_images, device)
                end = time.time()
                
                total_loss += loss
                time_forward += (end - start)
            
            # Update logits every 16 steps (to match original batch accumulation)
            if (step + 1) % 16 == 0:
                total_loss = total_loss / 16
                
                # Compute gradient signal
                start = time.time()
                grad_loss = total_loss - vanilla_loss
                delta = delta * 0.99 + grad_loss * 0.01
                
                # Update logits for all target modules
                for module in model.modules():
                    if hasattr(module, "logits"):
                        if len(module.logits.shape) == 4:  # Conv2d
                            # Flatten conv weights for easier processing
                            out_ch, in_ch, kh, kw = module.logits.shape
                            _logits_ = module.logits.view(out_ch, -1)
                            _mask_ = module.mask.view(out_ch, -1).float()
                        else:  # Linear
                            _logits_ = module.logits
                            _mask_ = module.mask.float()
                        
                        # Apply (2:4) grouping
                        flat_size = _logits_.shape[1]
                        M = 4
                        if flat_size % M != 0:
                            continue  # Skip layers that don't fit (2:4) pattern perfectly
                        
                        _logits_grouped = _logits_.view(_logits_.shape[0], -1, M)
                        _mask_grouped = _mask_.view(_mask_.shape[0], -1, M)
                        
                        _probs_ = torch.softmax(_logits_grouped, dim=2)
                        _probs_ = torch.clamp(_probs_, min=1e-8, max=1.0)
                        
                        # Compute policy gradient (simplified version)
                        R = 1 / (_mask_grouped / (1 - _probs_ + 1e-8)).sum(dim=2, keepdim=True)
                        _grad_log_probs_ = _mask_grouped / _probs_ + R * _mask_grouped / ((1 - _probs_)**2 + 1e-8)
                        
                        dot = (_grad_log_probs_ * _probs_).sum(dim=2, keepdim=True)
                        _grad_logit_ = (_probs_ * (_grad_log_probs_ - dot))
                        
                        # Reshape back and update
                        if len(module.logits.shape) == 4:  # Conv2d
                            _grad_logit_ = _grad_logit_.view(out_ch, in_ch, kh, kw)
                        else:  # Linear
                            _grad_logit_ = _grad_logit_.view(module.logits.shape)
                        
                        module.logits.data.copy_(module.logits.data - args.lr * (grad_loss - delta) * _grad_logit_)
                
                end = time.time()
                time_backward = end - start
                
                # Re-sample masks based on current logits
                start = time.time()
                for module in model.modules():
                    if hasattr(module, "logits"):
                        module.mask.data.copy_(generate_mask(module.logits))
                end = time.time()
                time_generate_mask = end - start
                
                # Normalize times
                time_forward = time_forward / 16
                
                # Logging
                improvement = vanilla_loss - total_loss
                global_step += 1
                
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Step {step+1} --> "
                    f"Loss: {total_loss:.6f}({vanilla_loss:.6f}) --> {improvement:.6f} | "
                    f"Delta (improvements): {(-delta):.6f}"
                )
                
                # SwanLab logging
                if use_swanlab:
                    swanlab.log({
                        "train/current_loss": total_loss,
                        "train/vanilla_loss": vanilla_loss,
                        "train/improvement": improvement,
                        "train/delta": -delta,
                        "train/grad_loss": grad_loss,
                        "train/epoch": epoch + 1,
                        "train/global_step": global_step,
                        "time/forward": time_forward,
                        "time/backward": time_backward,
                        "time/mask_generation": time_generate_mask
                    })
                    
                    # Log sparsity statistics every 10 steps
                    if global_step % 10 == 0:
                        total_params = 0
                        zero_params = 0
                        logits_stats = {"max": [], "min": [], "mean": [], "std": []}
                        
                        for module in model.modules():
                            if hasattr(module, "logits"):
                                # Collect logits statistics
                                logits_stats["max"].append(module.logits.max().item())
                                logits_stats["min"].append(module.logits.min().item())
                                logits_stats["mean"].append(module.logits.mean().item())
                                logits_stats["std"].append(module.logits.std().item())
                                
                            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                                weight = module.weight.data
                                total_params += weight.numel()
                                zero_params += (weight == 0).sum().item()
                        
                        current_sparsity = zero_params / total_params if total_params > 0 else 0.0
                        
                        swanlab.log({
                            "model/current_sparsity": current_sparsity,
                            "logits/max_mean": np.mean(logits_stats["max"]),
                            "logits/min_mean": np.mean(logits_stats["min"]),
                            "logits/mean_mean": np.mean(logits_stats["mean"]),
                            "logits/std_mean": np.mean(logits_stats["std"])
                        })
                
                loss_list.append(improvement)
                train_loss_list.append(total_loss)
                
                # Reset
                total_loss = 0
                time_forward = 0
                time_backward = 0
                time_generate_mask = 0
            
            if (step + 1) >= max_step:
                break
        
        if (step + 1) >= max_step:
            break
    
    # Save results
    output_dir = f"{args.output_dir}/lr{args.lr}_epoch{args.epoch}_logits{args.logits}_size{args.dataset_size}_diffusion/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Unwrap and save model - CRITICAL: This applies final masks to weights
    print("Applying final masks and saving model...")
    
    # Debug: Check if logits exist before unwrapping
    logits_count = 0
    for module in model.modules():
        if hasattr(module, "logits"):
            logits_count += 1
    print(f"Found {logits_count} modules with logits before unwrapping")
    
    logits_out = os.path.join(output_dir, "logits/")
    mask_unwrapper_diffusion(model, logits_out, args.save)
    
    # Debug: Check final model sparsity
    total_params = 0
    zero_params = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            weight = module.weight.data
            total_params += weight.numel()
            zero_params += (weight == 0).sum().item()
    
    final_sparsity = zero_params / total_params if total_params > 0 else 0.0
    print(f"Final model sparsity after unwrapping: {final_sparsity:.4f}")
    
    # Save model checkpoint AFTER unwrapping (so weights contain sparsity)
    checkpoint_dir = os.path.join(output_dir, "checkpoint")
    pipeline_to_save = DDPMPipeline(unet=model, scheduler=scheduler)
    pipeline_to_save.save_pretrained(checkpoint_dir)
    print(f"Model saved to {checkpoint_dir}")
    
    # Save training curves - ensure all tensors are moved to CPU
    loss_array = []
    train_loss_array = []
    
    for item in loss_list:
        if isinstance(item, torch.Tensor):
            loss_array.append(item.cpu().numpy())
        else:
            loss_array.append(item)
    
    for item in train_loss_list:
        if isinstance(item, torch.Tensor):
            train_loss_array.append(item.cpu().numpy())
        else:
            train_loss_array.append(item)
    
    np.save(os.path.join(output_dir, "loss_improvements.npy"), np.array(loss_array))
    np.save(os.path.join(output_dir, "loss_training.npy"), np.array(train_loss_array))
    
    # Save training summary - ensure all values are CPU-compatible and JSON serializable
    def to_cpu_value(val):
        if isinstance(val, torch.Tensor):
            return val.cpu().item() if val.numel() == 1 else val.cpu().numpy().tolist()
        elif isinstance(val, np.ndarray):
            return val.item() if val.size == 1 else val.tolist()
        elif isinstance(val, (np.integer, np.floating)):
            return val.item()
        return val
    
    summary = {
        'original_model': args.original_model,
        'mask_source': 'initial_mask_diffusion/ (from pruned model)',
        'methodology': 'ORIGINAL_MODEL + DYNAMIC_MASK',
        'dataset': args.dataset,
        'lr': float(args.lr),
        'epoch': int(args.epoch),
        'logits_magnitude': float(args.logits),
        'dataset_size': int(args.dataset_size),
        'batch_size': int(args.batch_size),
        'final_improvement': to_cpu_value(loss_array[-1]) if loss_array else 0,
        'mean_improvement': float(np.mean(loss_array)) if loss_array else 0,
        'std_improvement': float(np.std(loss_array)) if loss_array else 0,
        'note': 'Corrected implementation following original MaskPro methodology'
    }
    
    import json
    with open(os.path.join(output_dir, "training_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Training Completed ===")
    print(f"Output directory: {output_dir}")
    print(f"Final improvement: {to_cpu_value(loss_array[-1]):.6f}" if loss_array else "No improvements recorded")
    print(f"Mean improvement: {np.mean(loss_array):.6f}" if loss_array else "No improvements recorded")
    
    # SwanLab: Log final results
    if use_swanlab:
        final_stats = {
            "final/output_directory": output_dir,
            "final/total_steps": global_step,
            "final/final_improvement": to_cpu_value(loss_array[-1]) if loss_array else 0,
            "final/mean_improvement": np.mean(loss_array) if loss_array else 0,
            "final/std_improvement": np.std(loss_array) if loss_array else 0,
            "final/max_improvement": np.max(loss_array) if loss_array else 0,
            "final/min_improvement": np.min(loss_array) if loss_array else 0,
            "final/total_training_time": time.time() - start_time
        }
        
        # Final model sparsity
        total_params = 0
        zero_params = 0
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                weight = module.weight.data
                total_params += weight.numel()
                zero_params += (weight == 0).sum().item()
        
        final_sparsity = zero_params / total_params if total_params > 0 else 0.0
        final_stats["final/model_sparsity"] = final_sparsity
        
        swanlab.log(final_stats)
        
        # Log training curves as artifacts
        if loss_array:
            # Create improvement curve plot
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Loss improvement over time
            plt.subplot(2, 2, 1)
            plt.plot(loss_array)
            plt.title('Loss Improvement Over Time')
            plt.xlabel('Training Step')
            plt.ylabel('Improvement (Vanilla Loss - Current Loss)')
            plt.grid(True)
            
            # Plot 2: Cumulative improvement
            plt.subplot(2, 2, 2)
            cumulative_improvement = np.cumsum(loss_array)
            plt.plot(cumulative_improvement)
            plt.title('Cumulative Loss Improvement')
            plt.xlabel('Training Step')
            plt.ylabel('Cumulative Improvement')
            plt.grid(True)
            
            # Plot 3: Current loss over time
            plt.subplot(2, 2, 3)
            plt.plot(train_loss_array)
            plt.title('Current Loss Over Time')
            plt.xlabel('Training Step')
            plt.ylabel('Current Loss')
            plt.grid(True)
            
            # Plot 4: Moving average improvement
            plt.subplot(2, 2, 4)
            window_size = min(20, len(loss_array) // 4)
            if window_size > 1:
                moving_avg = np.convolve(loss_array, np.ones(window_size)/window_size, mode='valid')
                plt.plot(range(window_size-1, len(loss_array)), moving_avg)
                plt.title(f'Moving Average Improvement (window={window_size})')
                plt.xlabel('Training Step')
                plt.ylabel('Moving Average Improvement')
                plt.grid(True)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "training_curves.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log the plot to SwanLab
            swanlab.log({"training_curves": swanlab.Image(plot_path)})
        
        print(f"SwanLab experiment completed: {args.experiment_name}")
        swanlab.finish()
    
    print("Training completed successfully!")