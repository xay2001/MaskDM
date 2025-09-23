#!/usr/bin/env python3
"""
调试2:4稀疏加速问题
"""

import torch
import time
import numpy as np
from diffusers import DDIMPipeline, DDIMScheduler

def check_sparse_support():
    """检查A100稀疏支持情况"""
    print("🔍 检查A100硬件和软件稀疏支持...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"设备: {torch.cuda.get_device_name()}")
    
    # 检查A100特定要求
    device_name = torch.cuda.get_device_name()
    if "A100" in device_name:
        print("✅ A100硬件：支持2:4结构化稀疏")
    else:
        print(f"⚠️  非A100硬件({device_name})：可能不支持硬件稀疏加速")
    
    # 检查PyTorch版本
    pytorch_version = torch.__version__
    major, minor = pytorch_version.split('.')[:2]
    if int(major) >= 2 or (int(major) == 1 and int(minor) >= 12):
        print("✅ PyTorch版本：支持A100稀疏加速")
    else:
        print(f"⚠️  PyTorch版本过低({pytorch_version})：建议升级到1.12+")
    
    # 启用A100优化配置
    try:
        # 启用优化配置
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            print("✅ Flash SDP：已启用")
        
        if hasattr(torch.backends.cuda, 'allow_tf32'):
            torch.backends.cuda.allow_tf32 = True
            print("✅ TF32：已启用")
            
        # 检查Tensor Core支持
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
            print("✅ cuDNN TF32：已启用")
            
    except Exception as e:
        print(f"⚠️  优化配置设置失败: {e}")
    
    # 检查是否支持2:4稀疏
    device = torch.device("cuda")
    
    # 测试基本的稀疏张量支持
    try:
        # 创建一个简单的2:4稀疏模式
        dense = torch.randn(4, 4, device=device, dtype=torch.float16)  # 使用FP16
        # 创建2:4掩码 [1,1,0,0] 模式
        mask = torch.tensor([[1,1,0,0], [0,0,1,1], [1,0,1,0], [0,1,0,1]], 
                           dtype=torch.bool, device=device)
        sparse_tensor = dense * mask.float()
        
        print(f"✅ FP16稀疏张量操作：支持")
        print(f"稀疏度: {(sparse_tensor == 0).float().mean().item()*100:.1f}%")
        
        # 检查是否有SparseSemiStructuredTensor支持（PyTorch 2.0+）
        if hasattr(torch, 'sparse'):
            if hasattr(torch.sparse, 'SparseSemiStructuredTensor'):
                print("✅ SparseSemiStructuredTensor：支持")
                
                # 测试真正的稀疏张量创建
                try:
                    # 创建2:4稀疏张量
                    dense_weights = torch.randn(128, 128, device=device, dtype=torch.float16)
                    sparse_weights = torch.sparse.SparseSemiStructuredTensor.from_dense(dense_weights)
                    print("✅ 2:4稀疏张量创建：成功")
                except Exception as se:
                    print(f"⚠️  2:4稀疏张量创建失败: {se}")
            else:
                print("❌ SparseSemiStructuredTensor：不支持")
        else:
            print("❌ torch.sparse模块：不支持")
            
    except Exception as e:
        print(f"❌ 稀疏张量测试失败: {e}")

def test_different_configs():
    """测试不同配置的性能"""
    print("\n🧪 测试不同配置的性能...")
    
    dense_path = "/data/xay/MaskDM/pretrained/ddpm_ema_cifar10"
    pruned_path = "/data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2/pruned/unet_ema_pruned.pth"
    model_dir = "/data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2"
    
    configs = [
        {"dtype": torch.float32, "steps": 20, "name": "FP32_20步(基准)"},
        {"dtype": torch.float16, "steps": 20, "name": "FP16_20步(A100稀疏)"},
        {"dtype": torch.float16, "steps": 10, "name": "FP16_10步(快速测试)"},
        {"dtype": torch.float16, "steps": 50, "name": "FP16_50步(完整测试)"},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n🔸 测试配置: {config['name']}")
        
        try:
            # 加载Dense模型
            dense_pipeline = DDIMPipeline.from_pretrained(
                dense_path,
                torch_dtype=config["dtype"],
                use_safetensors=True
            )
            dense_pipeline = dense_pipeline.to('cuda')
            
            # 加载剪枝模型
            unet = torch.load(pruned_path, map_location='cpu')
            unet = unet.to('cuda', dtype=config["dtype"])
            
            pruned_pipeline = DDIMPipeline(
                unet=unet,
                scheduler=DDIMScheduler.from_pretrained(model_dir, subfolder="scheduler")
            )
            pruned_pipeline = pruned_pipeline.to('cuda')
            
            # 简单测试
            torch.cuda.synchronize()
            
            # Dense测试
            start_time = time.time()
            with torch.no_grad():
                _ = dense_pipeline(
                    batch_size=1,
                    num_inference_steps=config["steps"],
                    generator=torch.Generator(device='cuda').manual_seed(42)
                ).images
            torch.cuda.synchronize()
            dense_time = time.time() - start_time
            
            # 剪枝测试
            start_time = time.time()
            with torch.no_grad():
                _ = pruned_pipeline(
                    batch_size=1,
                    num_inference_steps=config["steps"],
                    generator=torch.Generator(device='cuda').manual_seed(42)
                ).images
            torch.cuda.synchronize()
            pruned_time = time.time() - start_time
            
            speedup = dense_time / pruned_time
            results[config['name']] = {
                'dense': dense_time,
                'pruned': pruned_time,
                'speedup': speedup
            }
            
            print(f"  Dense: {dense_time:.3f}s")
            print(f"  剪枝: {pruned_time:.3f}s")
            print(f"  加速比: {speedup:.2f}x")
            
            # 清理内存
            del dense_pipeline, pruned_pipeline, unet
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            results[config['name']] = None
    
    return results

def test_batch_sizes():
    """测试不同批量大小的影响"""
    print("\n🔢 测试不同批量大小...")
    
    dense_path = "/data/xay/MaskDM/pretrained/ddpm_ema_cifar10"
    pruned_path = "/data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2/pruned/unet_ema_pruned.pth"
    model_dir = "/data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2"
    
    batch_sizes = [1, 2, 4, 8]
    
    try:
        # 加载模型 (使用FP16，20步)
        dense_pipeline = DDIMPipeline.from_pretrained(
            dense_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        dense_pipeline = dense_pipeline.to('cuda')
        
        unet = torch.load(pruned_path, map_location='cpu')
        unet = unet.to('cuda', dtype=torch.float16)
        
        pruned_pipeline = DDIMPipeline(
            unet=unet,
            scheduler=DDIMScheduler.from_pretrained(model_dir, subfolder="scheduler")
        )
        pruned_pipeline = pruned_pipeline.to('cuda')
        
        for batch_size in batch_sizes:
            print(f"\n🔸 批量大小: {batch_size}")
            
            try:
                # Dense测试
                torch.cuda.synchronize()
                start_time = time.time()
                with torch.no_grad():
                    _ = dense_pipeline(
                        batch_size=batch_size,
                        num_inference_steps=20,
                        generator=torch.Generator(device='cuda').manual_seed(42)
                    ).images
                torch.cuda.synchronize()
                dense_time = time.time() - start_time
                
                # 剪枝测试
                torch.cuda.synchronize()
                start_time = time.time()
                with torch.no_grad():
                    _ = pruned_pipeline(
                        batch_size=batch_size,
                        num_inference_steps=20,
                        generator=torch.Generator(device='cuda').manual_seed(42)
                    ).images
                torch.cuda.synchronize()
                pruned_time = time.time() - start_time
                
                speedup = dense_time / pruned_time
                throughput_dense = batch_size / dense_time
                throughput_pruned = batch_size / pruned_time
                
                print(f"  Dense: {dense_time:.3f}s ({throughput_dense:.2f} imgs/s)")
                print(f"  剪枝: {pruned_time:.3f}s ({throughput_pruned:.2f} imgs/s)")
                print(f"  加速比: {speedup:.2f}x")
                
            except Exception as e:
                print(f"  ❌ 批量{batch_size}测试失败: {e}")
        
        # 清理
        del dense_pipeline, pruned_pipeline, unet
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ 批量测试失败: {e}")

def enable_a100_optimizations():
    """启用A100优化配置"""
    try:
        # 启用所有A100相关优化
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, 'allow_tf32'):
            torch.backends.cuda.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        print("✅ A100优化配置已启用")
    except Exception as e:
        print(f"⚠️  优化配置启用失败: {e}")

def main():
    print("🔧 A100稀疏加速调试工具")
    print("="*50)
    
    # 0. 启用A100优化
    enable_a100_optimizations()
    
    # 1. 检查基本支持
    check_sparse_support()
    
    # 2. 测试不同配置
    print("\n" + "="*50)
    config_results = test_different_configs()
    
    # 3. 测试批量大小
    print("\n" + "="*50)
    test_batch_sizes()
    
    # 4. 总结
    print("\n" + "="*50)
    print("📊 总结分析")
    print("="*50)
    
    if config_results:
        best_config = None
        best_speedup = 0
        
        for config_name, result in config_results.items():
            if result and result['speedup'] > best_speedup:
                best_speedup = result['speedup']
                best_config = config_name
        
        if best_config:
            print(f"✅ 最佳配置: {best_config}")
            print(f"✅ 最佳加速比: {best_speedup:.2f}x")
        else:
            print("❌ 未发现明显加速")
    
    print(f"\n💡 可能的改进建议:")
    print(f"  1. 尝试使用float16而不是float32")
    print(f"  2. 减少采样步数（20步而不是50步）")
    print(f"  3. 增加批量大小（如果内存允许）")
    print(f"  4. 检查PyTorch版本是否支持A100稀疏优化")
    print(f"  5. 确认模型的稀疏模式是否正确实现")

if __name__ == "__main__":
    main()
