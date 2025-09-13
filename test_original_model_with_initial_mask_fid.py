#!/usr/bin/env python3
"""
测试原始模型+初始mask的FID脚本
自动完成采样和FID计算的完整流程
"""

import os
import sys
import argparse
import time
import shutil
import torch
from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel, DDPMPipeline
from tqdm import tqdm
import accelerate

# 添加当前目录到路径，以便导入本地模块
sys.path.append('/data/xay/MaskDM')
sys.path.append('/data/xay/MaskDM/Maskpro')

# 导入mask wrapper功能
from Maskpro.wrapper_diffusion import mask_wrapper_diffusion

def setup_parser():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description="测试原始模型+初始mask的FID")
    
    # 模型相关参数
    parser.add_argument("--original_model", type=str, 
                       default="/data/xay/MaskDM/pretrained/ddpm_ema_cifar10",
                       help="原始DDPM模型路径")
    
    # 采样参数
    parser.add_argument("--total_samples", type=int, default=10000,
                       help="总采样数量 (默认10000用于FID计算)")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="批次大小")
    parser.add_argument("--ddim_steps", type=int, default=100,
                       help="DDIM采样步数")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    # 输出目录
    parser.add_argument("--output_dir", type=str, 
                       default="/data/xay/MaskDM/samples_original_with_initial_mask",
                       help="样本保存目录")
    
    # FID计算参数
    parser.add_argument("--reference_stats", type=str,
                       default="/data/xay/MaskDM/run/fid_stats_cifar10.npz",
                       help="参考数据集的FID统计文件")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备")
    
    # 控制选项
    parser.add_argument("--skip_sampling", action="store_true",
                       help="跳过采样，直接计算已有样本的FID")
    parser.add_argument("--skip_fid", action="store_true", 
                       help="只进行采样，跳过FID计算")
    parser.add_argument("--cleanup_samples", action="store_true",
                       help="计算完FID后删除样本文件")
    
    return parser

def load_model_with_initial_masks(model_path, device):
    """加载原始模型并应用初始mask"""
    print(f"正在加载原始模型: {model_path}")
    
    try:
        # 加载完整的DDPM pipeline
        if os.path.isfile(os.path.join(model_path, "model_index.json")):
            pipeline = DDPMPipeline.from_pretrained(model_path)
            unet = pipeline.unet.to(device)
            scheduler = pipeline.scheduler
            print(f"✓ 原始模型加载成功")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
    except Exception as e:
        print(f"❌ 加载原始模型失败: {e}")
        sys.exit(1)
    
    # 加载初始mask
    initial_mask_path = "/data/xay/MaskDM/Maskpro/initial_mask_diffusion"
    if not os.path.exists(initial_mask_path):
        print(f"❌ 初始mask目录不存在: {initial_mask_path}")
        sys.exit(1)
        
    initial_mask_files = [f.replace(".pt", "") for f in os.listdir(initial_mask_path) if f.endswith('.pt')]
    if not initial_mask_files:
        print(f"❌ 初始mask目录为空: {initial_mask_path}")
        sys.exit(1)
        
    print(f"找到 {len(initial_mask_files)} 个初始mask文件")
    
    # 应用初始mask到模型
    print("正在应用初始mask...")
    mask_wrapper_diffusion(
        unet, 
        initial_mask_files, 
        [],  # 没有learned masks
        logits_magnitude=10.0,
        targets=['down_blocks', 'up_blocks', 'mid_block']
    )
    print("✓ 初始mask应用完成")
    
    # 转换为DDIM pipeline以加快采样
    ddim_scheduler = DDIMScheduler.from_config(scheduler.config)
    ddim_pipeline = DDIMPipeline(unet=unet, scheduler=ddim_scheduler)
    
    return ddim_pipeline

def generate_samples(pipeline, args, device):
    """生成样本图像"""
    print(f"\n开始生成样本...")
    print(f"总样本数: {args.total_samples}")
    print(f"批次大小: {args.batch_size}")
    print(f"DDIM步数: {args.ddim_steps}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # 计算需要的批次数
    num_batches = args.total_samples // args.batch_size
    total_generated = num_batches * args.batch_size
    
    if total_generated < args.total_samples:
        print(f"注意: 由于批次大小限制，实际将生成 {total_generated} 个样本")
    
    # 生成样本
    sample_count = 0
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="生成样本批次"):
            try:
                # 生成一个批次的图像
                images = pipeline(
                    batch_size=args.batch_size,
                    num_inference_steps=args.ddim_steps,
                    generator=generator
                ).images
                
                # 保存图像
                for i, image in enumerate(images):
                    filename = os.path.join(args.output_dir, f"{sample_count:06d}.png")
                    image.save(filename)
                    sample_count += 1
                    
            except Exception as e:
                print(f"❌ 生成第 {batch_idx} 批次时出错: {e}")
                continue
    
    print(f"✓ 样本生成完成，共生成 {sample_count} 个样本")
    return sample_count

def calculate_fid(sample_dir, reference_stats, device):
    """计算FID分数"""
    print(f"\n开始计算FID...")
    print(f"样本目录: {sample_dir}")
    print(f"参考统计: {reference_stats}")
    
    # 检查文件是否存在
    if not os.path.exists(sample_dir):
        print(f"❌ 样本目录不存在: {sample_dir}")
        return None
        
    if not os.path.exists(reference_stats):
        print(f"❌ 参考统计文件不存在: {reference_stats}")
        return None
    
    # 计算样本数量
    sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"找到 {len(sample_files)} 个样本文件")
    
    if len(sample_files) == 0:
        print("❌ 样本目录为空")
        return None
    
    try:
        # 导入FID计算函数
        from fid_score import calculate_fid_given_paths
        
        # 计算FID
        fid_value = calculate_fid_given_paths(
            [sample_dir, reference_stats],
            batch_size=50,
            device=device,
            dims=2048,
            num_workers=4,
            num_samples=None,  # 使用所有样本
            res=None,  # 使用原始分辨率
            dataset_name=None
        )
        
        print(f"✓ FID计算完成")
        return fid_value
        
    except Exception as e:
        print(f"❌ FID计算失败: {e}")
        return None

def main():
    """主函数"""
    parser = setup_parser()
    args = parser.parse_args()
    
    print("=" * 60)
    print("测试原始模型+初始mask的FID")
    print("=" * 60)
    print(f"原始模型: {args.original_model}")
    print(f"输出目录: {args.output_dir}")
    print(f"参考统计: {args.reference_stats}")
    print(f"设备: {args.device}")
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    start_time = time.time()
    
    # 步骤1: 生成样本
    if not args.skip_sampling:
        print(f"\n{'='*20} 步骤1: 加载模型并生成样本 {'='*20}")
        
        # 加载模型
        pipeline = load_model_with_initial_masks(args.original_model, device)
        
        # 生成样本
        sample_count = generate_samples(pipeline, args, device)
        
        if sample_count == 0:
            print("❌ 没有成功生成任何样本")
            sys.exit(1)
            
        sampling_time = time.time() - start_time
        print(f"采样用时: {sampling_time:.2f}秒")
        
        # 清理GPU内存
        del pipeline
        torch.cuda.empty_cache()
    else:
        print(f"\n跳过采样步骤，使用现有样本: {args.output_dir}")
    
    # 步骤2: 计算FID
    if not args.skip_fid:
        print(f"\n{'='*20} 步骤2: 计算FID分数 {'='*20}")
        
        fid_start_time = time.time()
        fid_value = calculate_fid(args.output_dir, args.reference_stats, device)
        
        if fid_value is not None:
            fid_time = time.time() - fid_start_time
            print(f"FID计算用时: {fid_time:.2f}秒")
            
            print(f"\n{'='*20} 最终结果 {'='*20}")
            print(f"原始模型+初始mask的FID分数: {fid_value:.4f}")
            print(f"{'='*50}")
            
            # 保存结果到文件
            result_file = os.path.join(args.output_dir, "fid_result.txt")
            with open(result_file, 'w') as f:
                f.write(f"原始模型+初始mask的FID测试结果\n")
                f.write(f"{'='*40}\n")
                f.write(f"模型路径: {args.original_model}\n")
                f.write(f"样本数量: {args.total_samples}\n")
                f.write(f"DDIM步数: {args.ddim_steps}\n")
                f.write(f"随机种子: {args.seed}\n")
                f.write(f"FID分数: {fid_value:.4f}\n")
                f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"结果已保存到: {result_file}")
            
        else:
            print("❌ FID计算失败")
            sys.exit(1)
    else:
        print(f"\n跳过FID计算步骤")
    
    # 步骤3: 清理样本文件(可选)
    if args.cleanup_samples and os.path.exists(args.output_dir):
        print(f"\n清理样本文件...")
        # 只删除图像文件，保留结果文件
        for file in os.listdir(args.output_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                os.remove(os.path.join(args.output_dir, file))
        print(f"✓ 样本文件清理完成")
    
    total_time = time.time() - start_time
    print(f"\n总用时: {total_time:.2f}秒")
    print("测试完成!")

if __name__ == "__main__":
    main()


