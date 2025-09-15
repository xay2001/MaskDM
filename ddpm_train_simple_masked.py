#!/usr/bin/env python3
"""
简化版mask约束微调 - 基于原ddpm_train.py，只在优化器更新后应用mask约束
避免序列化问题，保持原有结构
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
    # 使用原ddpm_train.py的参数解析
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pruned_model_ckpt", type=str, default=None)

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-64",
        help="The output directory where the checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='./cache',
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
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
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
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
        "--save_model_steps", type=int, default=1000, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.0, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddim_num_inference_steps", type=int, default=100)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args

def load_masked_model_simple(model_path):
    """
    简化的mask模型加载，只加载权重和mask，不修改前向传播
    """
    print(f"正在加载剪枝模型: {model_path}")
    
    try:
        # 尝试标准加载
        if os.path.isdir(model_path):
            if os.path.exists(os.path.join(model_path, "model_index.json")):
                pipeline = DDPMPipeline.from_pretrained(model_path)
                unet = pipeline.unet
                scheduler = pipeline.scheduler
            else:
                unet = UNet2DModel.from_pretrained(model_path, subfolder="unet")
                scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        else:
            raise ValueError(f"无法识别的模型路径格式: {model_path}")
    except Exception as e:
        print(f"标准加载失败: {e}")
        print("使用自定义加载...")
        
        from safetensors.torch import load_file
        
        # 加载配置
        config_path = os.path.join(model_path, "unet", "config.json")
        unet = UNet2DModel.from_config(config_path)
        
        # 加载权重和mask
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
            print(f"✓ 加载了{len(model_state_dict)}个权重参数")
            
            # 注册mask为buffer
            mask_count = 0
            for mask_key, mask_value in mask_dict.items():
                layer_name = mask_key[:-5]  # 移除'.mask'
                layer = unet
                for part in layer_name.split('.'):
                    layer = getattr(layer, part)
                
                layer.register_buffer("mask", mask_value.bool())
                sparsity = (mask_value == 0).sum().item() / mask_value.numel() * 100
                print(f"✓ 注册mask: {layer_name} (稀疏度: {sparsity:.1f}%)")
                mask_count += 1
            
            print(f"✓ 总共注册了{mask_count}个mask")
        
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
    
    print(f"✅ 模型加载完成，参数量: {sum(p.numel() for p in unet.parameters()):,}")
    return unet, scheduler

def apply_mask_constraints(model):
    """
    应用mask约束 - 将被mask的权重置零
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and hasattr(module, 'mask'):
                # 应用mask约束
                module.weight.data = module.weight.data * module.mask.to(module.weight.dtype)

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
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")
    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

        # 设置wandb离线模式（避免网络问题）
        os.environ["WANDB_MODE"] = "offline"
        
        # 添加SwanLab同步
        try:
            import swanlab
            # 同步wandb到SwanLab，wandb设置为离线模式
            swanlab.sync_wandb(wandb_run=False)
            print("SwanLab同步已启用，wandb数据将同步到SwanLab")
        except ImportError:
            print("SwanLab未安装，跳过同步设置")

    # Make one log on every process with the configuration for debugging.
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

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 加载mask模型
    model, noise_scheduler = load_masked_model_simple(args.model_path)

    # Get the datasets
    dataset = utils.get_dataset(args.dataset)
    logger.info(f"Dataset size: {len(dataset)}")
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    num_epochs = math.ceil(args.num_iters / len(train_dataloader))

    # Create EMA for the model.
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

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Total optimization steps = {args.num_iters}")

    global_step = 0
    first_epoch = 0

    # save the shell command
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, 'run.sh'), 'w') as f:
            f.write('python ' + ' '.join(sys.argv))

    # setup dropout
    if args.dropout>0:
        utils.set_dropout(model, args.dropout)

    # generate images before training
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
        torchvision.utils.save_image(torch.from_numpy(images).permute([0, 3, 1, 2]), os.path.join(args.output_dir, 'vis', 'before_training.png'))
        del unet
        del pipeline

    accelerator.wait_for_everyone()    
    # Train!
    os.makedirs(os.path.join(args.output_dir, 'vis'), exist_ok=True)
    for epoch in range(first_epoch, num_epochs):
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            model.train()
            if isinstance(batch, (list, tuple)):
                clean_images = batch[0]
            else:
                clean_images = batch
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]

            # 时间步采样 - 使用原ddpm_train.py的方法
            timesteps = torch.randint(
                low=0, high=noise_scheduler.config.num_train_timesteps, size=(bsz // 2 + 1,)
            ).to(clean_images.device)
            timesteps = torch.cat([timesteps, noise_scheduler.config.num_train_timesteps - timesteps - 1], dim=0)[:bsz]

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                # Predict the noise residual
                model_output = model(noisy_images, timesteps).sample
                loss = (noise - model_output).square().sum(dim=(1, 2, 3)).mean(dim=0) 
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                
                # 关键步骤：在优化器更新后应用mask约束
                if accelerator.sync_gradients:
                    apply_mask_constraints(accelerator.unwrap_model(model))
                
            # Checks if the accelerator has performed an optimization step behind the scenes
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

            # Save the model & generate sample images 
            if global_step % args.save_model_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    # save the model
                    unet = accelerator.unwrap_model(model).eval()
                    unet.zero_grad()
                    os.makedirs(os.path.join(args.output_dir, 'pruned'), exist_ok=True)
                    torch.save(unet, os.path.join(args.output_dir, 'pruned', 'unet_pruned.pth'))
                    torch.save(unet, os.path.join(args.output_dir, 'pruned', 'unet_pruned-{}.pth'.format(global_step)))
                    if args.use_ema:
                        ema_model.store(unet.parameters())
                        ema_model.copy_to(unet.parameters())
                        torch.save(unet, os.path.join(args.output_dir, 'pruned', 'unet_ema_pruned.pth'))
                        torch.save(unet, os.path.join(args.output_dir, 'pruned', 'unet_ema_pruned-{}.pth'.format(global_step)))
                    pipeline = DDPMPipeline(
                        unet=unet,
                        scheduler=noise_scheduler,
                    )
                    pipeline.save_pretrained(args.output_dir) 

                    # generate images
                    logger.info("Sampling images...")
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
                    torchvision.utils.save_image(torch.from_numpy(images).permute([0, 3, 1, 2]), os.path.join(args.output_dir, 'vis', 'iter-{}.png'.format(global_step)))
                    # denormalize the images and save to tensorboard
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
            if global_step>args.num_iters:
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
