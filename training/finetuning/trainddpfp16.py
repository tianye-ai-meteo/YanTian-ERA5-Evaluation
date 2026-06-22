import time
import sys
import os
baseline_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(baseline_path)
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader #, DistributedSampler # 不再需要 DDP Sampler
from model import BaselineModel
from finetuning_dataset import BaselineDataset
from config import *
import torch.distributed as dist
from loss import DynamicWeightedMAELoss
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import setup_logger, setup_distributed, cleanup_distributed, save_checkpoint, load_checkpoint
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR # 不再需要 SequentialLR
from torch.cuda.amp import autocast, GradScaler
from timm.models.layers import DropPath
import torch.nn as nn
import numpy as np
from datetime import timedelta
import gc
"""
Author: lity
Date: 2025-06-01
Description: 
    文件功能描述: 启动8卡分布式训练脚本，使用step-wise的余弦衰减学习率
    文件启动方式描述: 
nohup torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 trainddpfp16.py >finetuningddpfp16.log 2>&1 &
"""

# -*- coding: utf-8 -*-
def train(continue_train=continue_train):
    # 设置日志
    logger = setup_logger()

    # 设置分布式训练环境
    setup_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == train_world_size, f"训练时使用的GPU数量与配置文件中的train_world_size不一致，请检查配置文件。"
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"cuda:{local_rank}")


    if rank == 0:
        logger.info(f"分布式训练环境初始化成功。")


    # 准备模型
    # 初始化模型时，它仍然是 float32    
    model = BaselineModel().to(device)

    if rank == 0:
        logger.info(f"{device}上模型初始化成功 (FP32)。")

    # --- 准备训练---

    # --- 定义损失函数 ---
    # 损失函数通常不需要手动转换类型，它们会根据输入张量的类型进行计算
    loss_fn_rmse = DynamicWeightedMAELoss().to(device) # 移动到设备
    if rank == 0:
        logger.info(f"{device}上RMSE损失函数创建成功 (FP32).")

    # --- 定义优化器 ---
    optimizer = optim.AdamW(model.parameters(), lr=new_lr, betas=(0.9, 0.95), weight_decay=0.1)
    if rank == 0:
        logger.info(f"{device}上优化器创建成功，初始峰值LR: {new_lr}。")


    # --- 初始化 GradScaler ---
    scaler = GradScaler(enabled=True) # enabled=True 表示启用混合精度
    if rank == 0:
        logger.info(f"{device}上GradScaler创建成功，用于混合精度训练。")


    start_epoch = 0 # 默认从 epoch 0 开始

    # --- 加载断点 ---
    # 处理继续训练的情况
    if continue_train:
        checkpoint_path = os.path.join(baseline_path, saved_model_dir, f'model_epoch_{model_epoch}.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Fine-tuning checkpoint not found: {checkpoint_path}. "
                "Set YANTIAN_FINETUNE_CONTINUE_TRAIN=false to start without a checkpoint, "
                "or set YANTIAN_FINETUNE_CHECKPOINT_DIR/model_epoch in config.py to an existing checkpoint."
            )
        next_epoch, missing, unexpected = load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=None,
            scheduler=None,
            map_location=device,   
            strict=True        
        )
        start_epoch = next_epoch
        if missing or unexpected:
            logger.warning(f"加载检查点时缺少参数: {missing}")
            logger.warning(f"加载检查点时发现意外参数: {unexpected}")
        else:
            logger.info(f"加载检查点成功，从 epoch {start_epoch} 继续训练。")

        if use_new_lr:
            optimizer = optim.AdamW(model.parameters(), lr=new_lr, betas=new_beta, weight_decay=new_weight_decay)
            if rank == 0:
                logger.info(f"使用新的学习率调度器。新学习率: {new_lr},, 新beta: {new_beta}, 新weight_decay: {new_weight_decay}")

        dist.barrier()
    # DDP 封装模型
    # 模型已经是 BF16 类型
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # 创建保存模型的目录
    if rank == 0:
        os.makedirs(os.path.join(baseline_path, saved_model_dir), exist_ok=True)
        os.makedirs(os.path.join(baseline_path, saved_picture_path), exist_ok=True)



    # 准备数据集
    train_dataset = BaselineDataset(start_year=train_start_year, end_year=train_end_year, step=4*days)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # shuffle=False 对于 sampler 后的 DataLoader 是推荐的，因为 sampler 已经处理了 shuffle
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True, shuffle=False, prefetch_factor=2,persistent_workers=False)
    if rank == 0:
        logger.info(f"数据集加载成功, 数据集大小: {len(train_loader)}")
        

    step_start_time = time.time()
    model.train() # 设置为训练模式
    optimizer.zero_grad()
    train_step =0
    for epoch in range(10):
        train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            train_step += 1
            if train_step > train_steps:
                break
            if rank == 0 and i == 0:
                start_time = time.time()

            data, target = batch
            
            #  循环学习整体天气过程
            for a in range(days):
                # 将对应的输入和标签转移到显卡中
                if a == 0:
                    input = data.to(device, non_blocking=True)
                else:
                    input = input.detach().to(device, non_blocking=True)
                for k in range(4*a, 4*(a+1)):
                    target[k] = target[k].to(device, non_blocking=True)


                with autocast(enabled=True, dtype=torch.float16): # 启用 autocast 上下文
                    total_loss = 0.0
                    total_MAE = 0.0
                    for j in range(4*a, 4*(a+1)):
                        output_norm = model(input) # 模型输出在 autocast 下可能是 FP16
                        loss_rmse = loss_fn_rmse(output_norm, target[j]) # 损失计算也在 autocast 上下文中
                        loss = loss_rmse 
                        total_loss += loss
                        total_MAE += loss_rmse.item()
                        # 构建新的输入
                        input_1 = output_norm.unsqueeze(1) # (B, 1, 69, H, W)
                        input_0 = input[:,1,:,:,:].unsqueeze(1) # (B, 1, 69, H, W)
                        input = torch.cat([input_0, input_1], dim=1) # (B, 2, 69, H, W)


                    loss = total_loss / 4
                    loss_MAE = total_MAE / 4

                # 检查 loss 是否为 NaN 或 Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f" Batch {i+1}: Loss is NaN or Inf! Skipping batch.")
                    # 在纯 BF16 下，数值不稳定性可能更常见，需要注意学习率和模型初始化
                    continue # 跳过这个 batch

                # --- 反向传播 (使用 scaler.scale) ---
                scaler.scale(loss).backward() # 使用 GradScaler 缩放损失并执行反向传播
                # --- 更新权重 (使用 scaler.step) ---
                scaler.step(optimizer) # GradScaler 负责 unscale 梯度并调用 optimizer.step()
                # --- 更新 scaler 的缩放因子 ---
                scaler.update() # 更新 GradScaler 的缩放因子
                # 梯度清零
                optimizer.zero_grad()

                if i % log_interval == 0 and rank == 0:
                    end_time = time.time()
                    logger.info(f"----------------------------------train_step {train_step} 完成，用时: {end_time - start_time:.2f}s, 第{a+1}天的平均MAE: {loss_MAE:.6f},  loss_dtype: {loss.dtype}, lr: {optimizer.param_groups[0]['lr']:.8f}----------------------------------")
                
            # --- 优化张量生命周期管理 ---
            del data, target
            if 'output_norm' in locals(): del output_norm
            del loss, loss_MAE, input, input_0, input_1
            torch.cuda.empty_cache()
            gc.collect()

            if train_step % 100 == 0 :
                # 保存 checkpoint 
                if rank == 0:
                    logger.info(f"train_step {train_step}: 保存 checkpoint...")
                    ckpt_path = os.path.join(baseline_path, saved_model_dir, f"model_step_{train_step}.pt")
                    save_checkpoint(
                        path=ckpt_path,
                        model=model,
                        optimizer=None,
                        scheduler=None,  # 保存调度器状态
                        epoch=None                  # 保存“已完成”的 epoch
                    )
                    logger.info(f"step {train_step}: 已保存到 {ckpt_path}")
                dist.barrier()

    # 计算并记录 epoch 平均训练损失
    step_end_time = time.time()
    step_duration = step_end_time - step_start_time


    # 记录日志，标明使用的损失类型
    if rank == 0:
        current_lr = optimizer.param_groups[0]['lr'] # 获取当前学习率
        logger.info(f'Train completed in {step_duration/3600:.2f}h. LR: {current_lr:.8f}')

    if rank == 0:
        logger.info("Training finished.")

    cleanup_distributed()

if __name__ == "__main__":
    # 可以在这里添加命令行参数解析来覆盖 config.py 中的设置
    train()
