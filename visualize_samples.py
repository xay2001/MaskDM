"""
生成扩散模型采样图片的网格可视化
用于论文展示，将一个batch的图片拼接在一起
"""
import os
import argparse
from PIL import Image
import numpy as np
import math


def create_image_grid(image_dir, num_images=64, grid_cols=8, output_path=None):
    """
    从目录中读取图片并创建网格布局
    
    Args:
        image_dir: 图片所在目录
        num_images: 要显示的图片数量（默认64，即一个batch）
        grid_cols: 网格的列数（默认8，则8x8的网格）
        output_path: 输出图片的路径
    
    Returns:
        拼接后的PIL Image对象
    """
    # 查找process_0子目录
    if os.path.exists(os.path.join(image_dir, 'process_0')):
        image_dir = os.path.join(image_dir, 'process_0')
    
    # 获取图片文件列表
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')],
                        key=lambda x: int(x.split('.')[0]))
    
    # 限制图片数量
    image_files = image_files[:num_images]
    actual_num = len(image_files)
    
    if actual_num == 0:
        raise ValueError(f"在目录 {image_dir} 中没有找到PNG图片")
    
    print(f"从 {image_dir} 读取 {actual_num} 张图片")
    
    # 读取第一张图片以获取尺寸
    first_image = Image.open(os.path.join(image_dir, image_files[0]))
    img_width, img_height = first_image.size
    
    # 计算网格尺寸
    grid_rows = math.ceil(actual_num / grid_cols)
    
    # 创建空白画布
    grid_width = img_width * grid_cols
    grid_height = img_height * grid_rows
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # 将图片放置到网格中
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path)
        
        # 计算位置
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * img_width
        y = row * img_height
        
        # 粘贴图片
        grid_image.paste(img, (x, y))
    
    # 保存图片
    if output_path:
        grid_image.save(output_path)
        print(f"网格图片已保存到: {output_path}")
    
    return grid_image


def main():
    parser = argparse.ArgumentParser(description='生成扩散模型采样图片的网格可视化')
    parser.add_argument('--input_dirs', type=str, nargs='+', required=True,
                        help='输入图片目录（可以指定多个）')
    parser.add_argument('--output_dir', type=str, default='./visualization_results',
                        help='输出目录')
    parser.add_argument('--num_images', type=int, default=64,
                        help='每个网格显示的图片数量（默认64）')
    parser.add_argument('--grid_cols', type=int, default=8,
                        help='网格的列数（默认8）')
    parser.add_argument('--output_names', type=str, nargs='+', default=None,
                        help='输出文件名（可选，默认使用输入目录名）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 为每个输入目录生成网格图
    for idx, input_dir in enumerate(args.input_dirs):
        # 确定输出文件名
        if args.output_names and idx < len(args.output_names):
            output_name = args.output_names[idx]
        else:
            # 使用输入目录的最后一部分作为文件名
            dir_name = os.path.basename(os.path.normpath(input_dir))
            output_name = f"{dir_name}_grid.png"
        
        output_path = os.path.join(args.output_dir, output_name)
        
        print(f"\n处理目录 {idx + 1}/{len(args.input_dirs)}: {input_dir}")
        
        try:
            create_image_grid(
                image_dir=input_dir,
                num_images=args.num_images,
                grid_cols=args.grid_cols,
                output_path=output_path
            )
        except Exception as e:
            print(f"处理 {input_dir} 时出错: {e}")
    
    print(f"\n所有网格图片已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()



