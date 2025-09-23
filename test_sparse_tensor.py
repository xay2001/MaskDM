#!/usr/bin/env python3
"""
测试SparseSemiStructuredTensor功能和真正的硬件加速
"""

import torch
import time
import numpy as np
from typing import Tuple

def check_sparse_support():
    """检查当前环境是否支持SparseSemiStructuredTensor"""
    print("🔍 检查稀疏张量支持情况")
    print("="*60)
    
    # 基本信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
    
    # 检查SparseSemiStructuredTensor支持
    try:
        if hasattr(torch.sparse, 'SparseSemiStructuredTensor'):
            print("✅ SparseSemiStructuredTensor: 支持")
            return True
        else:
            print("❌ SparseSemiStructuredTensor: 不支持")
            return False
    except Exception as e:
        print(f"❌ 检查稀疏张量时出错: {e}")
        return False

def create_24_sparse_weight(shape: Tuple[int, int], dtype=torch.float16) -> torch.Tensor:
    """创建完美的2:4稀疏权重矩阵"""
    weight = torch.zeros(shape, dtype=dtype)
    
    # 确保能被4整除
    total_elements = weight.numel()
    groups_of_4 = total_elements // 4
    
    # 将权重展平并重塑为(groups, 4)
    flat_weight = weight.view(-1)
    weight_groups = flat_weight[:groups_of_4 * 4].view(groups_of_4, 4)
    
    # 在每组4个元素中，前2个设为非零
    for i in range(groups_of_4):
        weight_groups[i, :2] = torch.randn(2, dtype=dtype) * 0.1
    
    return weight.contiguous()

def test_sparse_conversion():
    """测试稀疏张量转换功能"""
    print(f"\n🧪 测试稀疏张量转换")
    print("="*40)
    
    try:
        # 创建测试权重 - 确保形状适合2:4模式
        print("📦 创建2:4稀疏权重矩阵...")
        weight_shape = (256, 256)  # 总共65536个元素，能被4整除
        dense_weight = create_24_sparse_weight(weight_shape, torch.float16)
        
        print(f"权重形状: {dense_weight.shape}")
        print(f"权重稀疏度: {(dense_weight == 0).float().mean().item():.1%}")
        
        # 验证2:4模式
        flat_weight = dense_weight.view(-1)
        groups_of_4 = flat_weight.view(-1, 4)
        zero_counts = (groups_of_4 == 0).sum(dim=1)
        perfect_24_groups = (zero_counts == 2).sum().item()
        total_groups = groups_of_4.shape[0]
        compliance_rate = perfect_24_groups / total_groups
        
        print(f"2:4合规率: {compliance_rate:.1%}")
        
        if compliance_rate < 0.99:
            print("⚠️ 权重不是完美的2:4模式，创建完美的2:4权重...")
            # 重新创建确保完美2:4
            flat_weight = dense_weight.view(-1)
            groups_of_4 = flat_weight.view(-1, 4)
            # 强制每组前2个非零，后2个为零
            groups_of_4[:, :2] = torch.randn(groups_of_4.shape[0], 2, dtype=torch.float16) * 0.1
            groups_of_4[:, 2:] = 0
            dense_weight = flat_weight.view(weight_shape)
            
            # 重新验证
            zero_counts = (groups_of_4 == 0).sum(dim=1)
            perfect_24_groups = (zero_counts == 2).sum().item()
            compliance_rate = perfect_24_groups / total_groups
            print(f"修正后2:4合规率: {compliance_rate:.1%}")
        
        # 移到GPU
        if torch.cuda.is_available():
            dense_weight = dense_weight.cuda()
            print(f"✅ 权重已移至GPU")
        
        # 尝试转换为稀疏张量
        print("🔧 尝试转换为SparseSemiStructuredTensor...")
        
        # 设置强制使用CUTLASS
        if hasattr(torch.sparse, 'SparseSemiStructuredTensor'):
            torch.sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = True
            print("✅ 已启用CUTLASS加速")
        
        sparse_weight = torch.sparse.to_sparse_semi_structured(dense_weight)
        print("✅ 成功转换为SparseSemiStructuredTensor!")
        print(f"稀疏张量类型: {type(sparse_weight)}")
        print(f"稀疏张量形状: {sparse_weight.shape}")
        print(f"稀疏张量设备: {sparse_weight.device}")
        
        return dense_weight, sparse_weight
        
    except Exception as e:
        print(f"❌ 稀疏张量转换失败: {e}")
        print("💡 可能需要更新PyTorch版本或GPU不支持")
        return None, None

def benchmark_matmul_performance(dense_weight: torch.Tensor, sparse_weight: torch.Tensor):
    """对比密集矩阵和稀疏矩阵的运算性能"""
    print(f"\n⚡ 性能基准测试")
    print("="*40)
    
    if dense_weight is None or sparse_weight is None:
        print("❌ 无法进行基准测试，稀疏张量转换失败")
        return
    
    # 创建输入张量
    batch_size = 32
    input_size = dense_weight.shape[1]
    x = torch.randn(batch_size, input_size, dtype=torch.float16, device=dense_weight.device)
    
    # 预热
    print("🔥 GPU预热...")
    for _ in range(10):
        _ = torch.matmul(x, dense_weight.T)
        _ = torch.matmul(x, sparse_weight.T)
    
    torch.cuda.synchronize()
    
    # 测试密集矩阵运算
    print("📊 测试密集矩阵运算...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        result_dense = torch.matmul(x, dense_weight.T)
    
    torch.cuda.synchronize()
    dense_time = time.time() - start_time
    
    # 测试稀疏矩阵运算
    print("📊 测试稀疏矩阵运算...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(100):
        result_sparse = torch.matmul(x, sparse_weight.T)
    
    torch.cuda.synchronize()
    sparse_time = time.time() - start_time
    
    # 计算加速比
    speedup = dense_time / sparse_time
    
    print(f"\n📈 性能结果:")
    print(f"  密集矩阵时间: {dense_time:.4f}s")
    print(f"  稀疏矩阵时间: {sparse_time:.4f}s")
    print(f"  加速比: {speedup:.2f}x")
    
    # 验证结果一致性
    error = torch.max(torch.abs(result_dense - result_sparse)).item()
    print(f"  结果误差: {error:.6f}")
    
    if speedup > 1.1:
        print("🎉 获得了显著的硬件加速!")
    elif speedup > 1.0:
        print("✅ 获得了轻微的加速")
    else:
        print("⚠️ 没有获得加速，可能需要调整")
    
    return speedup

def test_model_weight_conversion():
    """测试将模型权重转换为稀疏格式"""
    print(f"\n🔄 测试模型权重转换")
    print("="*40)
    
    try:
        # 加载你的剪枝模型
        model_path = "/data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2/pruned/unet_ema_pruned.pth"
        print(f"📂 加载模型: {model_path}")
        
        model_state = torch.load(model_path, map_location='cpu')
        print("✅ 模型加载成功")
        
        converted_weights = {}
        conversion_count = 0
        
        for name, param in model_state.items():
            if 'weight' in name and param.dim() >= 2 and param.numel() > 1000:
                # 检查是否符合2:4模式
                flat_param = param.view(-1)
                if flat_param.numel() % 4 == 0:
                    groups_of_4 = flat_param.view(-1, 4)
                    zero_counts = (groups_of_4 == 0).sum(dim=1)
                    perfect_24_groups = (zero_counts == 2).sum().item()
                    total_groups = groups_of_4.shape[0]
                    compliance_rate = perfect_24_groups / total_groups
                    
                    if compliance_rate > 0.8:  # 至少80%符合2:4模式
                        try:
                            # 转换为FP16并移到GPU
                            weight_fp16 = param.to(torch.float16).cuda()
                            
                            # 尝试转换为稀疏张量
                            sparse_weight = torch.sparse.to_sparse_semi_structured(weight_fp16)
                            converted_weights[name] = {
                                'original': param,
                                'sparse': sparse_weight,
                                'compliance': compliance_rate
                            }
                            conversion_count += 1
                            print(f"✅ {name}: 转换成功 (合规率: {compliance_rate:.1%})")
                            
                        except Exception as e:
                            print(f"⚠️ {name}: 转换失败 - {e}")
        
        print(f"\n📊 转换统计:")
        print(f"  成功转换层数: {conversion_count}")
        print(f"  总权重层数: {len([n for n in model_state.keys() if 'weight' in n])}")
        
        if conversion_count > 0:
            print("🎉 部分模型权重成功转换为硬件加速格式!")
            return converted_weights
        else:
            print("❌ 没有权重能够转换为稀疏格式")
            return None
            
    except Exception as e:
        print(f"❌ 模型权重转换失败: {e}")
        return None

def main():
    print("🚀 SparseSemiStructuredTensor功能测试")
    print("="*60)
    
    # 检查支持情况
    if not check_sparse_support():
        print("\n❌ 当前环境不支持SparseSemiStructuredTensor")
        print("💡 需要PyTorch >= 2.1.0 且GPU支持稀疏运算")
        return
    
    # 测试稀疏张量转换
    dense_weight, sparse_weight = test_sparse_conversion()
    
    # 性能基准测试
    if dense_weight is not None and sparse_weight is not None:
        speedup = benchmark_matmul_performance(dense_weight, sparse_weight)
    
    # 测试模型权重转换
    converted_weights = test_model_weight_conversion()
    
    print(f"\n🎯 总结:")
    print("="*40)
    if dense_weight is not None:
        print("✅ SparseSemiStructuredTensor功能正常")
        if 'speedup' in locals() and speedup > 1.0:
            print(f"⚡ 获得了 {speedup:.2f}x 硬件加速")
        else:
            print("⚠️ 未获得明显加速")
    else:
        print("❌ SparseSemiStructuredTensor功能异常")
    
    if converted_weights and len(converted_weights) > 0:
        print(f"🔄 {len(converted_weights)}个模型权重已转换为稀疏格式")
    else:
        print("❌ 模型权重无法转换为稀疏格式")

if __name__ == "__main__":
    main()
