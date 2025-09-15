#!/usr/bin/env python3
"""
剪枝模型微调脚本
基于ddpm_train.py修改，专门用于微调已剪枝的扩散模型
"""

import argparse
import inspect
import logging
import math
import os, sys

import accelerate
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from packaging import version
from torchvision import transforms
import torchvision
from tqdm.auto import tqdm
import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, DDIMPipeline, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_accelerate_version, is_tensorboard_available, is_wandb_available

import utils

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="剪枝模型微调脚本")
    
    # 剪枝模型相关参数
    parser.add_argument("--pruned_model_path", type=str, required=True,
                       help="剪枝模型的路径")
    parser.add_argument("--load_masks", action="store_true", default=True,
                       help="是否加载剪枝masks")
    
    # 数据集参数
    parser.add_argument("--dataset", type=str, default="cifar10",
                       help="训练数据集")
    parser.add_argument("--dataset_config_name", type=str, default=None,
                       help="数据集配置名称")
    parser.add_argument("--train_data_dir", type=str, default=None,
                       help="训练数据目录")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="finetuned_pruned_model",
                       help="输出目录")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--cache_dir", type=str, default='./cache',
                       help="缓存目录")
    
    # 训练参数
    parser.add_argument("--resolution", type=int, default=32,
                       help="图像分辨率")
    parser.add_argument("--center_crop", default=False, action="store_true",
                       help="是否中心裁剪")
    parser.add_argument("--train_batch_size", type=int, default=16,
                       help="训练batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                       help="评估batch size")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                       help="数据加载器工作进程数")
    
    # 优化器参数 - 为微调调整默认值
    parser.add_argument("--num_iters", type=int, default=5000,
                       help="微调迭代次数（相比训练较少）")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="学习率（比训练时更小）")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                       help="学习率调度器")
    parser.add_argument("--lr_warmup_steps", type=int, default=100,
                       help="学习率预热步数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="梯度累积步数")
    
    # Adam优化器参数
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6,
                       help="权重衰减（微调时较小）")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    
    # EMA参数
    parser.add_argument("--use_ema", action="store_true", default=True,
                       help="是否使用EMA")
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=3/4)
    parser.add_argument("--ema_max_decay", type=float, default=0.999)
    
    # 保存和评估参数
    parser.add_argument("--save_model_steps", type=int, default=500,
                       help="模型保存间隔")
    parser.add_argument("--eval_steps", type=int, default=250,
                       help="评估间隔")
    
    # 扩散模型参数
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddim_num_inference_steps", type=int, default=100)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument("--prediction_type", type=str, default="epsilon",
                       choices=["epsilon", "sample"])
    
    # 日志参数
    parser.add_argument("--logger", type=str, default="tensorboard",
                       choices=["tensorboard", "wandb"])
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--use_swanlab", action="store_true", default=True,
                       help="是否使用SwanLab监控")
    
    # 分布式训练参数
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--mixed_precision", type=str, default="no",
                       choices=["no", "fp16", "bf16"])
    
    # 其他参数
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", 
                       action="store_true")
    
    args = parser.parse_args()
    
    # 验证参数
    if args.dataset is None and args.train_data_dir is None:
        raise ValueError("必须指定数据集名称或训练数据目录")
    
    return args

def load_pruned_model(model_path):
    """
    加载剪枝模型，处理mask相关的参数
    """
    print(f"正在加载剪枝模型: {model_path}")
    
    try:
        # 尝试直接加载整个pipeline
        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "model_index.json")):
            pipeline = DDPMPipeline.from_pretrained(model_path)
            unet = pipeline.unet
            scheduler = pipeline.scheduler
        # 如果只有unet文件夹
        elif os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "unet")):
            unet = UNet2DModel.from_pretrained(model_path, subfolder="unet")
            scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        # 如果是unet目录
        elif os.path.isdir(model_path) and "config.json" in os.listdir(model_path):
            unet = UNet2DModel.from_pretrained(model_path)
            # 创建默认scheduler
            scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear"
            )
        else:
            raise ValueError(f"无法识别的模型路径格式: {model_path}")
            
    except Exception as e:
        print(f"标准加载失败: {e}")
        print("尝试使用自定义加载方式...")
        
        # 自定义加载，跳过mask相关参数
        try:
            # 加载配置
            config_path = os.path.join(model_path, "config.json") if os.path.isdir(model_path) else model_path
            if not os.path.exists(config_path):
                config_path = os.path.join(model_path, "unet", "config.json")
            
            unet = UNet2DModel.from_config(config_path)
            
            # 加载权重，忽略mask相关参数
            weight_path = os.path.join(os.path.dirname(config_path), "diffusion_pytorch_model.safetensors")
            if not os.path.exists(weight_path):
                weight_path = os.path.join(os.path.dirname(config_path), "diffusion_pytorch_model.bin")
            
            if os.path.exists(weight_path):
                # 检查文件格式并相应加载
                if weight_path.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    state_dict = load_file(weight_path)
                else:
                    state_dict = torch.load(weight_path, map_location="cpu")
                
                # 过滤掉mask相关的参数
                filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('.mask')}
                unet.load_state_dict(filtered_state_dict, strict=False)
                print("成功加载模型权重（已过滤mask参数）")
            
            scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear"
            )
            
        except Exception as e2:
            raise ValueError(f"自定义加载也失败: {e2}")
    
    print(f"成功加载剪枝模型，参数量: {sum(p.numel() for p in unet.parameters()):,}")
    return unet, scheduler

def main(args):
    # 设置日志和加速器
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )
    
    # 初始化SwanLab监控
    swanlab_run = None
    if args.use_swanlab:
        try:
            import swanlab
            swanlab_run = swanlab.init(
                project="MaskDM-Finetune",
                experiment_name=f"pruned_model_finetune_{os.path.basename(args.pruned_model_path)}",
                description="剪枝模型微调实验",
                config={
                    "pruned_model_path": args.pruned_model_path,
                    "dataset": args.dataset,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.train_batch_size,
                    "num_iters": args.num_iters,
                    "use_ema": args.use_ema,
                    "lr_scheduler": args.lr_scheduler,
                    "mixed_precision": args.mixed_precision,
                }
            )
            print("✓ SwanLab监控已启用")
        except ImportError:
            print("⚠ SwanLab未安装，跳过SwanLab监控")
            swanlab_run = None
        except Exception as e:
            print(f"⚠ SwanLab初始化失败: {e}")
            swanlab_run = None
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # 创建输出目录
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载剪枝模型
    model, noise_scheduler = load_pruned_model(args.pruned_model_path)
    
    # 获取数据集
    dataset = utils.get_dataset(args.dataset)
    logger.info(f"数据集大小: {len(dataset)}")
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, 
        num_workers=args.dataloader_num_workers
    )
    num_epochs = math.ceil(args.num_iters / len(train_dataloader))
    
    # 创建EMA
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=False,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )
    
    # 创建优化器（使用较小的学习率）
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # 学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    
    # 准备训练
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    if args.use_ema:
        ema_model.to(accelerator.device)
    
    # 初始化追踪器
    if accelerator.is_main_process:
        accelerator.init_trackers("finetune_pruned_model")
    
    # 训练信息
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** 开始微调剪枝模型 *****")
    logger.info(f"  样本数量 = {len(dataset)}")
    logger.info(f"  每设备batch size = {args.train_batch_size}")
    logger.info(f"  总batch size = {total_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  训练轮数 = {num_epochs}")
    logger.info(f"  总优化步数 = {args.num_iters}")
    logger.info(f"  学习率 = {args.learning_rate}")
    
    # 保存运行命令
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, 'finetune_command.sh'), 'w') as f:
            f.write('python ' + ' '.join(sys.argv))
    
    # 微调前生成样本
    if accelerator.is_main_process:
        print("微调前生成样本...")
        unet = accelerator.unwrap_model(model).eval()
        pipeline = DDIMPipeline(
            unet=unet,
            scheduler=DDIMScheduler(num_train_timesteps=args.ddpm_num_steps)
        )
        pipeline.scheduler.set_timesteps(args.ddim_num_inference_steps)
        images = pipeline(
            batch_size=args.eval_batch_size,
            num_inference_steps=args.ddim_num_inference_steps,
            output_type="numpy",
        ).images
        
        os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
        torchvision.utils.save_image(
            torch.from_numpy(images).permute([0, 3, 1, 2]), 
            os.path.join(args.output_dir, 'samples', 'before_finetune.png')
        )
        del unet, pipeline
    
    accelerator.wait_for_everyone()
    
    # 开始微调
    global_step = 0
    first_epoch = 0
    
    for epoch in range(first_epoch, num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"微调轮次 {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            model.train()
            
            if isinstance(batch, (list, tuple)):
                clean_images = batch[0]
            else:
                clean_images = batch
                
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            
            # 时间步采样
            timesteps = torch.randint(
                low=0, high=noise_scheduler.config.num_train_timesteps, size=(bsz // 2 + 1,)
            ).to(clean_images.device)
            timesteps = torch.cat([timesteps, noise_scheduler.config.num_train_timesteps - timesteps - 1], dim=0)[:bsz]
            
            # 添加噪声
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                
                # 前向传播
                model_output = model(noisy_images, timesteps).sample
                loss = (noise - model_output).square().sum(dim=(1, 2, 3)).mean(dim=0)
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
            
            # 更新步数
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1
            
            # 记录日志
            logs = {
                "loss": loss.detach().item(), 
                "lr": lr_scheduler.get_last_lr()[0], 
                "step": global_step
            }
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            # SwanLab日志记录
            if swanlab_run is not None and global_step % 10 == 0:  # 每10步记录一次
                swanlab_run.log({
                    "train/loss": loss.detach().item(),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/step": global_step,
                    "train/epoch": epoch
                })
            
            # 评估和保存
            if global_step % args.eval_steps == 0:
                if accelerator.is_main_process:
                    print(f"步骤 {global_step}: 生成评估样本...")
                    unet = accelerator.unwrap_model(model).eval()
                    if args.use_ema:
                        ema_model.store(unet.parameters())
                        ema_model.copy_to(unet.parameters())
                    
                    pipeline = DDIMPipeline(
                        unet=unet,
                        scheduler=DDIMScheduler(num_train_timesteps=args.ddpm_num_steps)
                    )
                    pipeline.scheduler.set_timesteps(args.ddim_num_inference_steps)
                    images = pipeline(
                        batch_size=args.eval_batch_size,
                        num_inference_steps=args.ddim_num_inference_steps,
                        output_type="numpy",
                    ).images
                    
                    if args.use_ema:
                        ema_model.restore(unet.parameters())
                    
                    torchvision.utils.save_image(
                        torch.from_numpy(images).permute([0, 3, 1, 2]), 
                        os.path.join(args.output_dir, 'samples', f'step_{global_step}.png')
                    )
                    
                    # 记录到日志
                    images_processed = (images * 255).round().astype("uint8")
                    if args.logger == "tensorboard":
                        try:
                            if is_accelerate_version(">=", "0.17.0.dev0"):
                                tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                            else:
                                tracker = accelerator.get_tracker("tensorboard")
                            tracker.add_images("samples", images_processed.transpose(0, 3, 1, 2), global_step)
                        except Exception as e:
                            print(f"TensorBoard图像记录失败: {e}")
                    
                    # SwanLab记录生成的样本
                    if swanlab_run is not None:
                        import swanlab
                        swanlab_images = [swanlab.Image(img, caption=f"Generated at step {global_step}") 
                                        for img in images_processed[:8]]  # 只记录前8张图片
                        swanlab_run.log({
                            "generated_samples": swanlab_images,
                            "eval/step": global_step
                        })
                    
                    del unet, pipeline
            
            # 保存模型
            if global_step % args.save_model_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    print(f"保存模型检查点: 步骤 {global_step}")
                    unet = accelerator.unwrap_model(model).eval()
                    
                    # 保存pipeline
                    pipeline = DDPMPipeline(
                        unet=unet,
                        scheduler=noise_scheduler,
                    )
                    pipeline.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                    
                    # 保存EMA版本
                    if args.use_ema:
                        ema_model.store(unet.parameters())
                        ema_model.copy_to(unet.parameters())
                        pipeline_ema = DDPMPipeline(
                            unet=unet,
                            scheduler=noise_scheduler,
                        )
                        pipeline_ema.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{global_step}-ema"))
                        ema_model.restore(unet.parameters())
                    
                    del unet, pipeline
                    if args.use_ema:
                        del pipeline_ema
            
            if global_step >= args.num_iters:
                progress_bar.close()
                accelerator.wait_for_everyone()
                break
                
        if global_step >= args.num_iters:
            break
            
        progress_bar.close()
        accelerator.wait_for_everyone()
    
    # 保存最终模型
    if accelerator.is_main_process:
        print("保存最终微调模型...")
        unet = accelerator.unwrap_model(model).eval()
        pipeline = DDPMPipeline(
            unet=unet,
            scheduler=noise_scheduler,
        )
        pipeline.save_pretrained(args.output_dir)
        
        if args.use_ema:
            ema_model.store(unet.parameters())
            ema_model.copy_to(unet.parameters())
            pipeline_ema = DDPMPipeline(
                unet=unet,
                scheduler=noise_scheduler,
            )
            pipeline_ema.save_pretrained(os.path.join(args.output_dir, "final-ema"))
            ema_model.restore(unet.parameters())
    
    accelerator.end_training()
    
    # 完成SwanLab记录
    if swanlab_run is not None:
        swanlab_run.finish()
        print("✓ SwanLab记录已完成")
    
    print("微调完成！")

if __name__ == "__main__":
    args = parse_args()
    main(args)
