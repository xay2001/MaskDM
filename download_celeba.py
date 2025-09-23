#!/usr/bin/env python3
"""
CelebA-HQ数据集下载脚本
这个脚本会激活CelebA数据集的下载功能并下载数据
"""

import os
import sys
import argparse

# 添加ddpm_exp路径以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'ddpm_exp'))

from ddpm_exp.datasets.celeba import CelebA
import torchvision.transforms as transforms

def download_celeba_dataset(data_root="./data/celeba", download=True):
    """
    下载CelebA数据集
    
    Args:
        data_root (str): 数据保存路径
        download (bool): 是否下载数据集
    """
    print(f"正在下载CelebA数据集到: {data_root}")
    
    # 创建数据目录
    os.makedirs(data_root, exist_ok=True)
    
    # 临时修改CelebA类以启用下载
    print("正在准备下载CelebA数据集...")
    
    # 基本的数据变换
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        # 实例化数据集类，启用下载
        dataset = CelebA(
            root=data_root,
            split='train',
            target_type='attr',
            transform=transform,
            download=download  # 注意：需要手动修改celeba.py中被注释的下载代码
        )
        
        print(f"CelebA数据集下载完成!")
        print(f"训练集大小: {len(dataset)}")
        print(f"数据保存在: {data_root}")
        
        # 测试数据加载
        print("正在测试数据加载...")
        sample_image, sample_attr = dataset[0]
        print(f"样本图像尺寸: {sample_image.shape}")
        print(f"样本属性数量: {len(sample_attr)}")
        
        return True
        
    except Exception as e:
        print(f"下载过程中出现错误: {e}")
        print("请检查网络连接或手动激活ddpm_exp/datasets/celeba.py中的下载功能")
        return False

def main():
    parser = argparse.ArgumentParser(description='下载CelebA数据集')
    parser.add_argument('--data_root', type=str, default='./data/celeba',
                       help='数据保存路径 (默认: ./data/celeba)')
    parser.add_argument('--no_download', action='store_true',
                       help='不下载，仅检查现有数据')
    
    args = parser.parse_args()
    
    download_flag = not args.no_download
    
    if download_flag:
        print("=" * 60)
        print("CelebA数据集下载器")
        print("=" * 60)
        print("注意: 需要先手动激活下载功能!")
        print("请编辑 ddpm_exp/datasets/celeba.py 文件:")
        print("1. 取消注释第65-66行的下载代码")
        print("2. 取消注释第68-70行的完整性检查代码")
        print("=" * 60)
    
    success = download_celeba_dataset(args.data_root, download_flag)
    
    if success:
        print("\n✅ CelebA数据集准备完成!")
        print(f"数据位置: {os.path.abspath(args.data_root)}")
    else:
        print("\n❌ CelebA数据集下载失败!")
        print("请查看上面的错误信息并解决相关问题")

if __name__ == "__main__":
    main()
