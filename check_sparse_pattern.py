#!/usr/bin/env python3
"""
检查剪枝模型是否真的是2:4结构化稀疏
"""

import torch
import numpy as np
from diffusers import DDIMPipeline, DDIMScheduler

def check_24_pattern(weight_tensor, layer_name=""):
    """检查权重张量是否符合2:4结构化稀疏模式"""
    if len(weight_tensor.shape) < 2:
        return False, "权重维度不足2D"
    
    # 展平权重到2D
    if len(weight_tensor.shape) > 2:
        # 对于卷积层，重塑为 (out_channels, in_channels*kernel_size)
        original_shape = weight_tensor.shape
        if len(original_shape) == 4:  # Conv2d
            weight_2d = weight_tensor.reshape(original_shape[0], -1)
        else:
            weight_2d = weight_tensor.reshape(-1, original_shape[-1])
    else:
        weight_2d = weight_tensor
    
    # 检查是否可以按4个元素分组
    total_elements = weight_2d.numel()
    if total_elements % 4 != 0:
        return False, f"元素总数({total_elements})不能被4整除"
    
    # 重塑为 (N, 4) 格式以检查2:4模式
    try:
        # 将权重按行展平，然后重塑为每4个元素一组
        flat_weights = weight_2d.flatten()
        groups_of_4 = flat_weights.reshape(-1, 4)
        
        # 检查每组4个元素中是否恰好有2个为0
        zero_counts = (groups_of_4 == 0).sum(dim=1)
        
        # 统计符合2:4模式的组数
        valid_24_groups = (zero_counts == 2).sum().item()
        total_groups = groups_of_4.shape[0]
        
        # 计算2:4合规性
        compliance_rate = valid_24_groups / total_groups
        
        # 检查总稀疏度
        total_zeros = (flat_weights == 0).sum().item()
        sparsity_rate = total_zeros / total_elements
        
        return True, {
            "layer_name": layer_name,
            "total_groups": total_groups,
            "valid_24_groups": valid_24_groups,
            "compliance_rate": compliance_rate,
            "sparsity_rate": sparsity_rate,
            "is_perfect_24": compliance_rate == 1.0,
            "shape": weight_tensor.shape
        }
        
    except Exception as e:
        return False, f"检查过程出错: {e}"

def analyze_model_sparsity():
    """分析剪枝模型的稀疏模式"""
    print("🔍 检查剪枝模型的2:4结构化稀疏模式")
    print("="*60)
    
    # 加载剪枝模型
    pruned_path = "/data/xay/MaskDM/finetuned_results/masked_full_finetuned_v2/pruned/unet_ema_pruned.pth"
    
    try:
        print("📂 加载剪枝模型...")
        unet = torch.load(pruned_path, map_location='cpu')
        print("✅ 模型加载成功")
        
        total_params = 0
        total_zero_params = 0
        total_24_compliant_layers = 0
        total_layers_checked = 0
        
        print("\n📊 逐层分析稀疏模式:")
        print("-" * 80)
        print(f"{'层名称':<30} {'形状':<20} {'稀疏度':<10} {'2:4合规率':<12} {'状态'}")
        print("-" * 80)
        
        for name, param in unet.named_parameters():
            if 'weight' in name and param.numel() > 16:  # 只检查权重参数，跳过太小的层
                success, result = check_24_pattern(param.data, name)
                
                if success and isinstance(result, dict):
                    total_layers_checked += 1
                    layer_sparsity = result['sparsity_rate']
                    compliance = result['compliance_rate']
                    is_24 = result['is_perfect_24']
                    
                    # 累计统计
                    total_params += param.numel()
                    total_zero_params += (param.data == 0).sum().item()
                    
                    if is_24:
                        total_24_compliant_layers += 1
                    
                    # 状态标识
                    if is_24:
                        status = "✅ 完美2:4"
                    elif compliance > 0.9:
                        status = "🟡 接近2:4"
                    elif compliance > 0.5:
                        status = "🟠 部分2:4"
                    else:
                        status = "❌ 非2:4"
                    
                    print(f"{name:<30} {str(result['shape']):<20} {layer_sparsity:<10.1%} {compliance:<12.1%} {status}")
                else:
                    print(f"{name:<30} {'检查失败':<20} {'-':<10} {'-':<12} ❌")
        
        print("-" * 80)
        
        # 总体统计
        overall_sparsity = total_zero_params / total_params if total_params > 0 else 0
        layer_compliance_rate = total_24_compliant_layers / total_layers_checked if total_layers_checked > 0 else 0
        
        print(f"\n📈 总体分析:")
        print(f"  总参数数量: {total_params:,}")
        print(f"  零参数数量: {total_zero_params:,}")
        print(f"  总体稀疏度: {overall_sparsity:.2%}")
        print(f"  检查层数: {total_layers_checked}")
        print(f"  完美2:4层数: {total_24_compliant_layers}")
        print(f"  层级2:4合规率: {layer_compliance_rate:.1%}")
        
        print(f"\n🎯 结论:")
        if layer_compliance_rate > 0.8:
            print("✅ 模型是真正的2:4结构化稀疏！")
            print("💡 应该能够获得A100硬件加速")
        elif layer_compliance_rate > 0.5:
            print("🟡 模型部分符合2:4结构")
            print("⚠️  可能无法充分利用A100硬件加速")
        else:
            print("❌ 模型不是真正的2:4结构化稀疏")
            print("🔧 这解释了为什么没有获得硬件加速")
            
        # 检查是否需要转换为真正的稀疏张量
        print(f"\n🔧 优化建议:")
        if layer_compliance_rate > 0.8:
            print("1. 模型已经是2:4结构，但可能需要转换为SparseSemiStructuredTensor格式")
            print("2. 确保使用支持A100稀疏的PyTorch版本 (>=1.12)")
            print("3. 启用相关的CUDA优化配置")
        else:
            print("1. 需要重新进行2:4结构化剪枝")
            print("2. 确保剪枝过程严格按照2:4模式进行")
            print("3. 考虑使用专门的2:4稀疏训练工具")
        
        return overall_sparsity, layer_compliance_rate
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None, None

def check_sparse_tensor_conversion():
    """检查是否可以转换为真正的稀疏张量"""
    print(f"\n🧪 测试稀疏张量转换能力")
    print("="*40)
    
    try:
        # 检查PyTorch稀疏支持
        pytorch_version = torch.__version__
        print(f"PyTorch版本: {pytorch_version}")
        
        if hasattr(torch, 'sparse') and hasattr(torch.sparse, 'SparseSemiStructuredTensor'):
            print("✅ SparseSemiStructuredTensor: 支持")
            
            # 测试创建稀疏张量
            try:
                # 创建一个完美的2:4稀疏权重矩阵
                test_weight = torch.zeros(8, 8, dtype=torch.float16)
                # 按2:4模式填充
                for i in range(0, 8, 4):
                    for j in range(8):
                        test_weight[j, i:i+2] = torch.randn(2)  # 每4个中前2个非零
                
                print(f"测试权重稀疏度: {(test_weight == 0).float().mean().item():.1%}")
                
                # 尝试转换为稀疏张量
                sparse_weight = torch.sparse.SparseSemiStructuredTensor.from_dense(test_weight)
                print("✅ 稀疏张量转换: 成功")
                
                # 测试运算
                input_tensor = torch.randn(8, 8, dtype=torch.float16)
                result = torch.mm(sparse_weight.to_dense(), input_tensor)
                print("✅ 稀疏张量运算: 成功")
                
                return True
                
            except Exception as e:
                print(f"❌ 稀疏张量操作失败: {e}")
                return False
        else:
            print("❌ SparseSemiStructuredTensor: 不支持")
            return False
            
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def main():
    print("🔧 2:4结构化稀疏模式检查工具")
    print("="*60)
    
    # 分析模型稀疏模式
    sparsity, compliance = analyze_model_sparsity()
    
    # 检查稀疏张量转换能力
    can_convert = check_sparse_tensor_conversion()
    
    # 最终建议
    print(f"\n💡 最终诊断和建议:")
    print("="*40)
    
    if sparsity is not None and compliance is not None:
        if compliance > 0.8 and can_convert:
            print("🎉 你的模型是真正的2:4结构化稀疏！")
            print("🔧 下一步: 需要将权重转换为SparseSemiStructuredTensor格式")
            print("⚡ 转换后应该能获得A100硬件加速")
        elif compliance > 0.8:
            print("✅ 模型结构正确，但缺少稀疏张量支持")
            print("🔧 建议升级PyTorch到支持A100稀疏的版本")
        else:
            print("❌ 模型不是真正的2:4结构化稀疏")
            print("🔧 这就是为什么没有获得加速的原因！")
            print("💡 需要重新进行正确的2:4结构化剪枝")

if __name__ == "__main__":
    main()
