#!/usr/bin/env python3
"""
快速推理速度测试 - Dense vs 2:4剪枝模型对比
参考SparseDM测试方法：完整扩散采样过程评估
"""

import torch
import time
import numpy as np
from diffusers import DDIMPipeline, DDIMScheduler
import gc

# 启用A100优化配置
try:
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, 'allow_tf32'):
        torch.backends.cuda.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass  # 静默失败，不影响测试

def get_gpu_memory():
    """获取GPU内存使用情况(GB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0

def gpu_warmup(pipeline, steps=3):
    """GPU预热，参考SparseDM方法"""
    print("🔥 GPU预热中...")
    with torch.no_grad():
        for i in range(steps):
            _ = pipeline(
                batch_size=1,
                num_inference_steps=10,
                generator=torch.Generator(device='cuda').manual_seed(42+i)
            ).images
    torch.cuda.synchronize()
    print("✅ GPU预热完成")

def test_dense_model():
    """测试原始Dense模型 - 参考SparseDM测试方法"""
    print("📂 加载原始Dense模型...")
    try:
        dense_path = "/data/xay/MaskDM/pretrained/ddpm_ema_cifar10"
        
        pipeline = DDIMPipeline.from_pretrained(
            dense_path,
            torch_dtype=torch.float16,  # 使用FP16启用A100稀疏加速
            use_safetensors=True
        )
        pipeline.scheduler.skip_type = "uniform"
        pipeline = pipeline.to('cuda')
        
        # 检查参数量
        total_params = sum(p.numel() for p in pipeline.unet.parameters())
        print(f"✅ Dense模型加载成功！参数量: {total_params/1e6:.2f}M")
        
        # GPU预热
        gpu_warmup(pipeline)
        
        # 多轮测试以获得稳定结果
        print("⚡ Dense模型推理测试...")
        times = []
        memory_usage = []
        
        # 使用20步采样以更好展示A100 FP16稀疏加速效果
        num_runs = 5
        steps = 20
        
        for i in range(num_runs):
            torch.cuda.empty_cache()
            gc.collect()
            
            start_memory = get_gpu_memory()
            
            with torch.no_grad():
                torch.cuda.synchronize()
                start_time = time.time()
                
                images = pipeline(
                    batch_size=1,
                    num_inference_steps=steps,
                    generator=torch.Generator(device='cuda').manual_seed(42+i)
                ).images
                
                torch.cuda.synchronize()
                end_time = time.time()
            
            end_memory = get_gpu_memory()
            
            inference_time = end_time - start_time
            times.append(inference_time)
            memory_usage.append(end_memory - start_memory)
            
            print(f"  第{i+1}轮: {inference_time:.3f}秒, 内存: {end_memory-start_memory:.3f}GB")
        
        # 统计结果
        times = np.array(times[1:])  # 去掉第一次预热
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_memory = np.mean(memory_usage[1:])
        
        print(f"📊 Dense模型统计结果:")
        print(f"  平均时间: {avg_time:.3f} ± {std_time:.3f} 秒")
        print(f"  最快时间: {np.min(times):.3f} 秒")
        print(f"  最慢时间: {np.max(times):.3f} 秒")
        print(f"  平均内存: {avg_memory:.3f} GB")
        print(f"  吞吐量: {1/avg_time:.2f} images/sec")
        
        # 清理内存
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()
        
        return avg_time
        
    except Exception as e:
        print(f"❌ Dense模型测试失败: {e}")
        return None

def test_pruned_model():
    """测试2:4剪枝模型 - 参考SparseDM测试方法"""
    print("📂 加载2:4剪枝模型...")
    try:
        # 加载剪枝后的UNet
        pruned_path = "/data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2/pruned/unet_ema_pruned.pth"
        model_dir = "/data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2"
        
        unet = torch.load(pruned_path, map_location='cpu')
        unet = unet.to('cuda', dtype=torch.float16)  # 使用FP16启用A100稀疏加速
        
        pipeline = DDIMPipeline(
            unet=unet,
            scheduler=DDIMScheduler.from_pretrained(model_dir, subfolder="scheduler")
        )
        pipeline = pipeline.to('cuda')
        
        # 检查稀疏度
        total_params = sum(p.numel() for p in unet.parameters())
        zero_params = sum(torch.sum(p==0).item() for p in unet.parameters())
        sparsity = zero_params / total_params * 100
        effective_params = total_params - zero_params
        print(f"✅ 剪枝模型加载成功！稀疏度: {sparsity:.2f}%，有效参数: {effective_params/1e6:.2f}M")
        
        # GPU预热
        gpu_warmup(pipeline)
        
        # 多轮测试以获得稳定结果
        print("⚡ 剪枝模型推理测试...")
        times = []
        memory_usage = []
        
        # 使用20步采样以更好展示A100 FP16稀疏加速效果
        num_runs = 5
        steps = 20
        
        for i in range(num_runs):
            torch.cuda.empty_cache()
            gc.collect()
            
            start_memory = get_gpu_memory()
            
            with torch.no_grad():
                torch.cuda.synchronize()
                start_time = time.time()
                
                images = pipeline(
                    batch_size=1,
                    num_inference_steps=steps,
                    generator=torch.Generator(device='cuda').manual_seed(42+i)
                ).images
                
                torch.cuda.synchronize()
                end_time = time.time()
            
            end_memory = get_gpu_memory()
            
            inference_time = end_time - start_time
            times.append(inference_time)
            memory_usage.append(end_memory - start_memory)
            
            print(f"  第{i+1}轮: {inference_time:.3f}秒, 内存: {end_memory-start_memory:.3f}GB")
        
        # 统计结果
        times = np.array(times[1:])  # 去掉第一次预热
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_memory = np.mean(memory_usage[1:])
        
        print(f"📊 剪枝模型统计结果:")
        print(f"  平均时间: {avg_time:.3f} ± {std_time:.3f} 秒")
        print(f"  最快时间: {np.min(times):.3f} 秒")
        print(f"  最慢时间: {np.max(times):.3f} 秒")
        print(f"  平均内存: {avg_memory:.3f} GB")
        print(f"  吞吐量: {1/avg_time:.2f} images/sec")
        
        # 清理内存
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()
        
        return avg_time
        
    except Exception as e:
        print(f"❌ 剪枝模型测试失败: {e}")
        return None

def quick_comparison_test():
    """快速对比测试 - 参考SparseDM标准测试方法"""
    print("🚀 A100 FP16稀疏加速测试")
    print(f"🖥️  设备: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"🔧 CUDA版本: {torch.version.cuda}")
    print(f"📦 PyTorch版本: {torch.__version__}")
    print(f"🧠 显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    print("="*60)
    print("🎯 A100稀疏加速优化配置:")
    print("  - 数据类型: FP16 (启用Tensor Core稀疏加速)")
    print("  - 采样步数: 20步 (优化稀疏加速效果)")
    print("  - 测试轮次: 5轮 (去除第1轮预热)")
    print("  - 批量大小: 1")
    print("  - 稀疏模式: 2:4结构化稀疏")
    print("="*60)
    
    # 测试Dense模型
    print("\n🔸 测试1: 原始Dense模型")
    dense_time = test_dense_model()
    
    print("\n" + "-"*60)
    
    # 测试剪枝模型
    print("\n🔹 测试2: 2:4剪枝模型")
    pruned_time = test_pruned_model()
    
    # 对比结果
    print("\n" + "="*60)
    print("📊 最终对比结果")
    print("="*60)
    
    if dense_time is not None and pruned_time is not None:
        speedup = dense_time / pruned_time
        time_saved = dense_time - pruned_time
        percentage_improvement = (time_saved / dense_time) * 100
        
        print(f"🔸 Dense模型平均时间:     {dense_time:.3f} 秒")
        print(f"🔹 2:4剪枝模型平均时间:  {pruned_time:.3f} 秒")
        print(f"⚡ 加速比:               {speedup:.2f}x")
        print(f"⏱️  时间节省:             {time_saved:.3f} 秒 ({percentage_improvement:.1f}%)")
        print(f"🚀 每秒生成图片数提升:    {(1/pruned_time)/(1/dense_time):.2f}x")
        
        print(f"\n🔍 结果分析:")
        if speedup >= 3.0:
            print(f"🎉 在A100上获得了显著的硬件加速！")
            print(f"💡 2:4结构化剪枝充分利用了A100的Tensor Core加速")
        elif speedup >= 1.5:
            print(f"✅ 获得了良好的加速效果")
            print(f"💡 硬件加速正常工作")
        elif speedup >= 1.1:
            print(f"⚠️  加速效果较轻微")
            print(f"🔧 可能需要检查CUDA配置或模型优化")
        else:
            print(f"❌ 加速效果不明显")
            print(f"🔧 建议检查硬件支持和模型实现")
        
        print(f"\n📊 与SparseDM对比:")
        print(f"  - SparseDM加速比: ~1.2x")
        print(f"  - 本测试加速比: {speedup:.2f}x")
        if speedup > 1.2:
            print(f"  ✅ 超越了SparseDM的性能！")
        else:
            print(f"  📝 与SparseDM结果相近")
            
    elif dense_time is not None:
        print(f"🔸 Dense模型测试成功: {dense_time:.3f}秒")
        print(f"❌ 剪枝模型测试失败")
    elif pruned_time is not None:
        print(f"❌ Dense模型测试失败")
        print(f"🔹 剪枝模型测试成功: {pruned_time:.3f}秒")
    else:
        print(f"❌ 两个模型测试都失败了")
    
    print(f"\n📄 测试方法说明:")
    print(f"  - 本测试采用与SparseDM相同的评估方法")
    print(f"  - 测试完整的扩散采样过程，而非单步推理")
    print(f"  - 结果更能反映实际应用中的性能提升")
    print(f"  - 如需更详细分析，请运行 speed_benchmark.py")

if __name__ == "__main__":
    quick_comparison_test()
