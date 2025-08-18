import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def generate_mask_conv2d(logits, N=2, M=4):
    """Generate (N:M) sparse mask for Conv2d layers"""
    # For Conv2d: (out_channels, in_channels, kernel_h, kernel_w)
    out_ch, in_ch, kh, kw = logits.shape
    
    # Flatten to 2D: (out_channels, in_channels * kernel_h * kernel_w)
    logits_flat = logits.view(out_ch, -1)
    
    # Ensure dimensions are compatible with (N:M) pattern
    flat_size = logits_flat.shape[1]
    if flat_size % M != 0:
        # Pad to make it divisible by M
        pad_size = M - (flat_size % M)
        logits_flat = F.pad(logits_flat, (0, pad_size), value=-float('inf'))
    
    # Apply (N:M) pattern with numerical stability
    groups = logits_flat.view(out_ch, -1, M)  # (out_ch, groups, M)
    
    # Clamp extreme values to prevent overflow/underflow
    groups = torch.clamp(groups, min=-50, max=50)
    
    probs = torch.softmax(groups, dim=2)
    probs = torch.clamp(probs, min=1e-8, max=1.0)
    
    # Check for problematic values and fix them
    if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
        # Reset problematic groups to uniform distribution
        problematic_mask = torch.isnan(probs).any(dim=2) | torch.isinf(probs).any(dim=2) | (probs < 0).any(dim=2)
        probs[problematic_mask] = 1.0 / M
        
    # Ensure probabilities sum to 1 for each group
    probs = probs / probs.sum(dim=2, keepdim=True)
    
    # Sample N elements from each group of M
    sampled_indices = torch.multinomial(probs.view(-1, M), num_samples=N, replacement=False)
    mask = torch.zeros_like(groups.view(-1, M), dtype=torch.bool)
    mask.scatter_(1, sampled_indices, True)
    
    # Reshape back to conv2d shape
    mask = mask.view(out_ch, -1)
    if flat_size % M != 0:
        mask = mask[:, :flat_size]  # Remove padding
    
    return mask.view(out_ch, in_ch, kh, kw)

def generate_mask_linear(logits, N=2, M=4):
    """Generate (N:M) sparse mask for Linear layers"""
    out_features, in_features = logits.shape
    
    # Ensure dimensions are compatible with (N:M) pattern
    if in_features % M != 0:
        # Pad to make it divisible by M
        pad_size = M - (in_features % M)
        logits = F.pad(logits, (0, pad_size), value=-float('inf'))
    
    groups = logits.view(out_features, -1, M)
    
    # Clamp extreme values to prevent overflow/underflow  
    groups = torch.clamp(groups, min=-50, max=50)
    
    probs = torch.softmax(groups, dim=2)
    probs = torch.clamp(probs, min=1e-8, max=1.0)
    
    # Check for problematic values and fix them
    if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
        # Reset problematic groups to uniform distribution
        problematic_mask = torch.isnan(probs).any(dim=2) | torch.isinf(probs).any(dim=2) | (probs < 0).any(dim=2)
        probs[problematic_mask] = 1.0 / M
        
    # Ensure probabilities sum to 1 for each group
    probs = probs / probs.sum(dim=2, keepdim=True)
    
    sampled_indices = torch.multinomial(probs.view(-1, M), num_samples=N, replacement=False)
    mask = torch.zeros_like(groups.view(-1, M), dtype=torch.bool)
    mask.scatter_(1, sampled_indices, True)
    
    mask = mask.view(out_features, -1)
    if in_features % M != 0:
        mask = mask[:, :in_features]  # Remove padding
    
    return mask.view(out_features, in_features)

def generate_mask(logits, N=2, M=4):
    """Generate mask based on layer type"""
    if len(logits.shape) == 4:  # Conv2d
        return generate_mask_conv2d(logits, N, M)
    elif len(logits.shape) == 2:  # Linear
        return generate_mask_linear(logits, N, M)
    else:
        raise ValueError(f"Unsupported logits shape: {logits.shape}")

def generate_mask_from_logits_conv2d(logits, N=2, M=4):
    """Generate deterministic mask from logits for Conv2d"""
    out_ch, in_ch, kh, kw = logits.shape
    logits_flat = logits.view(out_ch, -1)
    
    flat_size = logits_flat.shape[1]
    if flat_size % M != 0:
        pad_size = M - (flat_size % M)
        logits_flat = F.pad(logits_flat, (0, pad_size), value=-float('inf'))
    
    groups = logits_flat.view(out_ch, -1, M)
    topk = torch.topk(groups, k=N, dim=2)
    topk_indices = topk.indices
    
    mask = torch.zeros_like(groups, dtype=torch.bool)
    mask.scatter_(2, topk_indices, True)
    
    mask = mask.view(out_ch, -1)
    if flat_size % M != 0:
        mask = mask[:, :flat_size]
    
    return mask.view(out_ch, in_ch, kh, kw)

def generate_mask_from_logits_linear(logits, N=2, M=4):
    """Generate deterministic mask from logits for Linear"""
    out_features, in_features = logits.shape
    
    if in_features % M != 0:
        pad_size = M - (in_features % M)
        logits = F.pad(logits, (0, pad_size), value=-float('inf'))
    
    groups = logits.view(out_features, -1, M)
    topk = torch.topk(groups, k=N, dim=2)
    topk_indices = topk.indices
    
    mask = torch.zeros_like(groups, dtype=torch.bool)
    mask.scatter_(2, topk_indices, True)
    
    mask = mask.view(out_features, -1)
    if in_features % M != 0:
        mask = mask[:, :in_features]
    
    return mask.view(out_features, in_features)

def generate_mask_from_logits(logits, N=2, M=4):
    """Generate deterministic mask from logits based on layer type"""
    if len(logits.shape) == 4:  # Conv2d
        return generate_mask_from_logits_conv2d(logits, N, M)
    elif len(logits.shape) == 2:  # Linear
        return generate_mask_from_logits_linear(logits, N, M)
    else:
        raise ValueError(f"Unsupported logits shape: {logits.shape}")

def mask_wrapper_diffusion(module, initial_mask_name_list, learned_mask_name_list, logits_magnitude, targets, prefix=""):
    """Wrap DDPM model with mask functionality"""
    
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        # Skip output layers and embeddings
        if not any(skip in full_name for skip in ["conv_out", "time_emb", "class_emb"]):
            mask_wrapper_diffusion(child, initial_mask_name_list, learned_mask_name_list, logits_magnitude, targets, full_name)
            
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        # Convert prefix to underscore format for matching with mask names
        prefix_underscore = prefix.replace(".", "_")
        if prefix_underscore in initial_mask_name_list:
            print(f"|--> loading mask for {prefix}, shape: {module.weight.shape}")
            mask_path = f"initial_mask_diffusion/{prefix.replace('.', '_')}.pt"
            try:
                mask = torch.load(mask_path, weights_only=True, map_location=module.weight.device).bool().contiguous()
                module.register_buffer("mask", mask)

                def forward_with_mask(self, x, *args, **kwargs):
                    masked_weight = self.weight * self.mask.to(self.weight.dtype)
                    if isinstance(self, nn.Conv2d):
                        return F.conv2d(x, masked_weight, self.bias, self.stride, 
                                      self.padding, self.dilation, self.groups)
                    else:  # Linear
                        return F.linear(x, masked_weight, self.bias)
                
                # Bind the method to the module
                import types
                module.forward = types.MethodType(forward_with_mask, module)
                        
                # Check for learned mask
                learned_mask_path = f"learned_mask_diffusion/{prefix.replace('.', '_')}.pt"
                if prefix_underscore in learned_mask_name_list and os.path.exists(learned_mask_path):
                    print(f"  |--> overwriting with learned mask for {prefix}")
                    learned_mask = torch.load(learned_mask_path, weights_only=True, map_location=module.weight.device).bool().contiguous()
                    module.mask.data.copy_(learned_mask)
                    # No need to rebind forward - already done above
                    
                # Add trainable logits for target layers
                # If targets is empty or contains "all", optimize all layers with masks
                should_optimize = False
                if not targets or targets == [""] or "all" in targets:
                    should_optimize = True  # Optimize all layers with masks
                elif any(target.replace(".", "_") in prefix_underscore for target in targets):
                    should_optimize = True  # Optimize layers matching targets (handle dot notation)
                
                if should_optimize:
                    print(f"    |--> generating logits for {prefix}")
                    # 恢复原始的简单初始化，不添加随机噪声
                    logits_init = (module.mask * logits_magnitude).float()
                    module.register_buffer("logits", logits_init)
                    # No need to rebind forward - already done above
                    
            except FileNotFoundError:
                print(f"Warning: Mask file not found for {prefix}")

def mask_unwrapper_diffusion(module, out_dir, save_mask, prefix=""):
    """Unwrap DDPM model and save final masks"""
    
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if not any(skip in full_name for skip in ["conv_out", "time_emb", "class_emb"]):
            mask_unwrapper_diffusion(child, out_dir, save_mask, full_name)
            
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if hasattr(module, "logits"):
            # Generate final mask from logits
            final_mask = generate_mask_from_logits(module.logits)
            module.mask.data.copy_(final_mask)
            
            if save_mask:
                # Save learned mask
                os.makedirs("learned_mask_diffusion", exist_ok=True)
                mask_save_path = f"learned_mask_diffusion/{prefix.replace('.', '_')}.pt"
                torch.save(module.mask, mask_save_path)
                print(f"Saved learned mask: {mask_save_path}")
            
            # Apply final mask to weights (重要：保持权重稀疏性)
            module.weight.data *= module.mask.data.to(module.weight.data.dtype)
            
            # 不要删除mask buffer，保留用于模型保存
            # 只清理logits buffer
            if hasattr(module, "_buffers") and "logits" in module._buffers:
                del module._buffers["logits"]
            
            print(f"Applied final mask to {prefix}, sparsity: {(module.mask == 0).float().mean():.4f}")
        
        elif hasattr(module, "mask"):
            # 对于只有mask没有logits的层，也应用mask到权重
            module.weight.data *= module.mask.data.to(module.weight.data.dtype)
            print(f"Applied mask to {prefix}, sparsity: {(module.mask == 0).float().mean():.4f}")