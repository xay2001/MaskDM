import torch
import torch.nn as nn
import os
from diffusers import DDPMPipeline
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the pruned DDPM model')
parser.add_argument('--output_dir', type=str, default='Maskpro/initial_mask_diffusion', help='Directory to save masks')
args = parser.parse_args()

def extract_masks_from_pruned_ddpm(model, output_dir):
    """Extract binary masks from a pruned DDPM model and save them"""
    os.makedirs(output_dir, exist_ok=True)
    
    extracted_count = 0
    total_layers = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total_layers += 1
            
            # Skip protected layers that shouldn't be masked for MaskPro
            if ('conv_in' in name or 
                'conv_out' in name or 
                'time_emb' in name or
                'class_emb' in name):
                print(f"Skipping protected layer: {name}")
                continue
            
            # Extract mask (True for non-zero weights, False for zero weights)
            weight = module.weight.data
            mask = (weight != 0)
            
            # Calculate sparsity
            sparsity = (weight == 0).float().mean().item()
            
            if sparsity > 0:  # Only save masks for layers that have been pruned
                # Clean the name for file saving (replace dots and slashes)
                clean_name = name.replace('.', '_').replace('/', '_')
                save_path = os.path.join(output_dir, f"{clean_name}.pt")
                
                torch.save(mask, save_path)
                print(f"Saved mask for {name} -> {clean_name}.pt (sparsity: {sparsity:.4f}, shape: {mask.shape})")
                extracted_count += 1
            else:
                print(f"Skipping {name} (no pruning detected, sparsity: {sparsity:.4f})")
    
    print(f"\n=== Mask Extraction Summary ===")
    print(f"Total layers examined: {total_layers}")
    print(f"Masks extracted: {extracted_count}")
    print(f"Output directory: {output_dir}")
    
    return extracted_count

if __name__ == '__main__':
    print(f"Loading DDPM model from {args.model_path}")
    
    # Try to load as DDPMPipeline first
    try:
        if os.path.isfile(os.path.join(args.model_path, "model_index.json")):
            # Load as pipeline
            pipeline = DDPMPipeline.from_pretrained(args.model_path)
            model = pipeline.unet
            print("Loaded as DDPM Pipeline")
        elif os.path.isfile(os.path.join(args.model_path, "pruned", "unet_pruned.pth")):
            # Load as saved PyTorch model
            model = torch.load(os.path.join(args.model_path, "pruned", "unet_pruned.pth"), map_location='cpu')
            print("Loaded as PyTorch model from pruned directory")
        else:
            raise FileNotFoundError("Could not find a valid DDPM model")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check the model path and format")
        exit(1)
    
    print(f"Model loaded successfully: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Extract masks
    num_masks = extract_masks_from_pruned_ddpm(model, args.output_dir)
    
    if num_masks == 0:
        print("\nWarning: No masks were extracted. This might indicate:")
        print("1. The model hasn't been pruned")
        print("2. All pruned layers were skipped due to protection rules")
        print("3. The pruning was not weight-level pruning")
    else:
        print(f"\nMask extraction completed! {num_masks} masks saved to {args.output_dir}")
        print("You can now use these masks for DDPM MaskPro training.")
        
        # Create a summary file
        summary_file = os.path.join(args.output_dir, "extraction_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"DDPM Mask Extraction Summary\n")
            f.write(f"Source model: {args.model_path}\n")
            f.write(f"Total masks extracted: {num_masks}\n")
            import datetime
            f.write(f"Extraction date: {datetime.datetime.now()}\n")
        
        print(f"Summary saved to {summary_file}")