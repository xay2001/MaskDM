#!/usr/bin/env python3
"""
修改版ddmp_train.py，支持mask剪枝模型的微调
关键特性：
1. 加载带mask的剪枝模型
2. 微调时保持mask约束（被mask的权重不更新）
3. 前向传播时应用mask
"""

import argparse
import inspect
import logging
import math
import os, sys

import accelerate
import torch
import torch.nn as nn
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
    parser = argparse.ArgumentParser(description="支持mask的剪枝模型微调脚本")
    parser.add_argument("--pruned_model_ckpt", type=str, default=None)

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集名称",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="数据集配置名称",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="剪枝模型路径",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="训练数据目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-masked-model",
        help="输出目录",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='./cache',
        help="缓存目录",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="图像分辨率",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="是否中心裁剪",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="训练batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="评估batch size"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="数据加载器工作进程数",
    )
    parser.add_argument(
        "--checkpoint_id",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--load_ema",
        action="store_true",
        default=False,
    )
    parser.add_argument("--num_iters", type=int, default=10000)
    parser.add_argument(
        "--save_model_steps", type=int, default=1000, help="模型保存间隔"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="学习率",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="学习率调度器",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="学习率预热步数"
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.0, help="权重衰减"
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="是否使用EMA",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=3 / 4)
    parser.add_argument("--ema_max_decay", type=float, default=0.999)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true"
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddim_num_inference_steps", type=int, default=100)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset is None and args.train_data_dir is None:
        raise ValueError("必须指定数据集名称或训练数据目录")

    return args

def apply_masks_to_model(model):
    """
    为模型中有mask的层添加mask约束的前向传播
    """
    def add_masked_forward(module, name):
        if hasattr(module, 'mask'):
            original_forward = module.forward
            
            def masked_forward(x, *args, **kwargs):
                # 在前向传播时应用mask
                if hasattr(module, 'weight') and hasattr(module, 'mask'):
                    # 临时应用mask
                    original_weight = module.weight.data.clone()
                    module.weight.data = module.weight.data * module.mask.to(module.weight.dtype)
                    
                    # 执行前向传播
                    if isinstance(module, nn.Conv2d):
                        result = F.conv2d(x, module.weight, module.bias, module.stride, 
                                        module.padding, module.dilation, module.groups)
                    elif isinstance(module, nn.Linear):
                        result = F.linear(x, module.weight, module.bias)
                    else:
                        result = original_forward(x, *args, **kwargs)
                    
                    # 恢复原始权重（重要：保持梯度计算正确）
                    module.weight.data = original_weight
                    return result
                else:
                    return original_forward(x, *args, **kwargs)
            
            module.forward = masked_forward
            print(f"✓ 为层 {name} 添加了mask约束前向传播")
    
    # 递归处理所有层
    for name, module in model.named_modules():
        add_masked_forward(module, name)

def apply_mask_constraints_to_gradients(model):
    """
    在梯度更新后应用mask约束，确保被mask的权重保持为0
    """
    def mask_gradients_hook(module, grad_input, grad_output):
        if hasattr(module, 'weight') and hasattr(module, 'mask'):
            # 将被mask的位置的梯度置零
            if module.weight.grad is not None:
                module.weight.grad.data = module.weight.grad.data * module.mask.to(module.weight.grad.dtype)
    
    # 为所有有mask的层注册hook
    for name, module in model.named_modules():
        if hasattr(module, 'mask'):
            module.register_backward_hook(mask_gradients_hook)
            print(f"✓ 为层 {name} 注册了mask梯度约束")

def load_masked_model(model_path):
    """
    加载带mask的剪枝模型
    """
    print(f"正在加载剪枝模型: {model_path}")
    
    try:
        # 尝试加载整个pipeline
        if os.path.exists(os.path.join(model_path, "model_index.json")):
            pipeline = DDPMPipeline.from_pretrained(model_path)
            unet = pipeline.unet
            scheduler = pipeline.scheduler
        else:
            # 分别加载unet和scheduler
            unet = UNet2DModel.from_pretrained(model_path, subfolder="unet")
            scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    except Exception as e:
        print(f"标准加载失败: {e}")
        print("尝试自定义加载...")
        
        # 自定义加载
        from safetensors.torch import load_file
        
        # 加载配置
        config_path = os.path.join(model_path, "unet", "config.json")
        unet = UNet2DModel.from_config(config_path)
        
        # 加载权重（包括mask）
        weight_path = os.path.join(model_path, "unet", "diffusion_pytorch_model.safetensors")
        if not os.path.exists(weight_path):
            weight_path = os.path.join(model_path, "unet", "diffusion_pytorch_model.bin")
        
        if os.path.exists(weight_path):
            if weight_path.endswith('.safetensors'):
                state_dict = load_file(weight_path)
            else:
                state_dict = torch.load(weight_path, map_location="cpu")
            
            # 分离权重和mask
            model_state_dict = {}
            mask_dict = {}
            
            for key, value in state_dict.items():
                if key.endswith('.mask'):
                    mask_dict[key] = value
                else:
                    model_state_dict[key] = value
            
            # 加载模型权重
            unet.load_state_dict(model_state_dict, strict=False)
            
            # 注册mask缓冲区
            for mask_key, mask_value in mask_dict.items():
                # 获取对应的层
                layer_name = mask_key[:-5]  # 移除'.mask'
                layer = unet
                for part in layer_name.split('.'):
                    layer = getattr(layer, part)
                
                # 注册mask
                layer.register_buffer("mask", mask_value.bool())
                print(f"✓ 为层 {layer_name} 注册了mask，稀疏度: {(mask_value == 0).sum().item() / mask_value.numel() * 100:.1f}%")
        
        # 创建scheduler
        scheduler_config_path = os.path.join(model_path, "scheduler", "scheduler_config.json")
        if os.path.exists(scheduler_config_path):
            scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        else:
            scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear"
            )
    
    # 应用mask约束
    apply_masks_to_model(unet)
    apply_mask_constraints_to_gradients(unet)
    
    print(f"✅ 成功加载剪枝模型，参数量: {sum(p.numel() for p in unet.parameters()):,}")
    return unet, scheduler

def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("请安装tensorboard")
    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("请安装wandb")
        import wandb
        os.environ["WANDB_MODE"] = "offline"
        
        try:
            import swanlab
            swanlab.sync_wandb(wandb_run=False)
            print("SwanLab同步已启用")
        except ImportError:
            print("SwanLab未安装，跳过同步设置")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 加载剪枝模型
    model, noise_scheduler = load_masked_model(args.model_path)

    # 获取数据集
    dataset = utils.get_dataset(args.dataset)
    logger.info(f"数据集大小: {len(dataset)}")
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
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

    # 初始化优化器
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
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    logger.info("***** 开始微调剪枝模型 *****")
    logger.info(f"  样本数量 = {len(dataset)}")
    logger.info(f"  每设备batch size = {args.train_batch_size}")
    logger.info(f"  总batch size = {total_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  训练轮数 = {num_epochs}")
    logger.info(f"  总优化步数 = {args.num_iters}")

    global_step = 0
    first_epoch = 0

    # 保存命令
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, 'run.sh'), 'w') as f:
            f.write('python ' + ' '.join(sys.argv))

    # setup dropout
    if args.dropout > 0:
        utils.set_dropout(model, args.dropout)

    # 微调前生成样本
    if accelerator.is_main_process:
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
        os.makedirs(os.path.join(args.output_dir, 'vis'), exist_ok=True)
        torchvision.utils.save_image(torch.from_numpy(images).permute([0, 3, 1, 2]), 
                                   os.path.join(args.output_dir, 'vis', 'before_finetune.png'))
        del unet
        del pipeline

    accelerator.wait_for_everyone()
    
    # 开始训练
    os.makedirs(os.path.join(args.output_dir, 'vis'), exist_ok=True)
    for epoch in range(first_epoch, num_epochs):
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
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
                # 前向传播（mask会自动应用）
                model_output = model(noisy_images, timesteps).sample
                loss = (noise - model_output).square().sum(dim=(1, 2, 3)).mean(dim=0)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                
                # 关键：在优化器更新后应用mask约束
                if accelerator.sync_gradients:
                    with torch.no_grad():
                        for name, module in accelerator.unwrap_model(model).named_modules():
                            if hasattr(module, 'weight') and hasattr(module, 'mask'):
                                # 确保被mask的权重保持为0
                                module.weight.data = module.weight.data * module.mask.to(module.weight.dtype)

            # 更新步数
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # 保存模型并生成样本
            if global_step % args.save_model_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    # 保存模型 - 清理hook避免序列化问题
                    unet = accelerator.unwrap_model(model).eval()
                    unet.zero_grad()
                    
                    # 临时移除hook进行保存
                    hooks = []
                    for module in unet.modules():
                        if hasattr(module, '_backward_hooks'):
                            hooks.extend(module._backward_hooks.values())
                            module._backward_hooks.clear()
                    
                    os.makedirs(os.path.join(args.output_dir, 'pruned'), exist_ok=True)
                    torch.save(unet, os.path.join(args.output_dir, 'pruned', 'unet_pruned.pth'))
                    torch.save(unet, os.path.join(args.output_dir, 'pruned', f'unet_pruned-{global_step}.pth'))
                    if args.use_ema:
                        ema_model.store(unet.parameters())
                        ema_model.copy_to(unet.parameters())
                        torch.save(unet, os.path.join(args.output_dir, 'pruned', 'unet_ema_pruned.pth'))
                        torch.save(unet, os.path.join(args.output_dir, 'pruned', f'unet_ema_pruned-{global_step}.pth'))
                    pipeline = DDPMPipeline(
                        unet=unet,
                        scheduler=noise_scheduler,
                    )
                    pipeline.save_pretrained(args.output_dir)

                    # 生成样本
                    logger.info("生成样本...")
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
                    torchvision.utils.save_image(torch.from_numpy(images).permute([0, 3, 1, 2]), 
                                               os.path.join(args.output_dir, 'vis', f'iter-{global_step}.png'))
                    
                    # 记录到日志
                    images_processed = (images * 255).round().astype("uint8")
                    if args.logger == "tensorboard":
                        try:
                            if is_accelerate_version(">=", "0.17.0.dev0"):
                                tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                            else:
                                tracker = accelerator.get_tracker("tensorboard")
                            tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), global_step)
                        except Exception as e:
                            print(f"TensorBoard记录失败: {e}")
                    elif args.logger == "wandb":
                        try:
                            accelerator.get_tracker("wandb").log(
                                {"test_samples": [wandb.Image(img) for img in images_processed], "steps": global_step},
                                step=global_step,
                            )
                        except Exception as e:
                            print(f"WandB记录失败: {e}")
                    del unet
                    del pipeline
            if global_step > args.num_iters:
                progress_bar.close()
                accelerator.wait_for_everyone()
                accelerator.end_training()
                return
        progress_bar.close()
        accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
