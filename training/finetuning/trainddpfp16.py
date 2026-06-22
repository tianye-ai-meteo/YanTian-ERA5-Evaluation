"""Distributed mixed-precision fine-tuning entry point for YanTian."""

import time
import sys
import os
baseline_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(baseline_path)
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader 
from model import BaselineModel
from finetuning_dataset import BaselineDataset
from config import *
import torch.distributed as dist
from loss import DynamicWeightedMAELoss
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import setup_logger, setup_distributed, cleanup_distributed, save_checkpoint, load_checkpoint
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR 
from torch.cuda.amp import autocast, GradScaler
from timm.models.layers import DropPath
import torch.nn as nn
import numpy as np
from datetime import timedelta
import gc

# -*- coding: utf-8 -*-
def train(continue_train=continue_train):
    
    logger = setup_logger()

    
    setup_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == train_world_size, f"The number of GPUs used for training does not match train_world_size in the configuration. Please check the configuration."
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"cuda:{local_rank}")


    if rank == 0:
        logger.info("Distributed training environment initialized successfully.")


    
    
    model = BaselineModel().to(device)

    if rank == 0:
        logger.info(f"Model initialized successfully on {device} (FP32).")

    

    
    
    loss_fn_rmse = DynamicWeightedMAELoss().to(device) 
    if rank == 0:
        logger.info(f"RMSE loss function created successfully on {device} (FP32).")

    
    optimizer = optim.AdamW(model.parameters(), lr=new_lr, betas=(0.9, 0.95), weight_decay=0.1)
    if rank == 0:
        logger.info(f"Optimizer created successfully on {device}; initial peak LR: {new_lr}.")


    
    scaler = GradScaler(enabled=True) 
    if rank == 0:
        logger.info(f"GradScaler created successfully on {device} for mixed-precision training.")


    start_epoch = 0 

    
    
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
            logger.warning(f"Missing parameters while loading checkpoint: {missing}")
            logger.warning(f"Unexpected parameters while loading checkpoint: {unexpected}")
        else:
            logger.info(f"Checkpoint loaded successfully; resuming from epoch {start_epoch}.")

        if use_new_lr:
            optimizer = optim.AdamW(model.parameters(), lr=new_lr, betas=new_beta, weight_decay=new_weight_decay)
            if rank == 0:
                logger.info(f"Using a new learning-rate scheduler. New LR: {new_lr}, new beta: {new_beta}, new weight_decay: {new_weight_decay}")

        dist.barrier()
    
    
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    
    if rank == 0:
        os.makedirs(os.path.join(baseline_path, saved_model_dir), exist_ok=True)
        os.makedirs(os.path.join(baseline_path, saved_picture_path), exist_ok=True)



    
    train_dataset = BaselineDataset(start_year=train_start_year, end_year=train_end_year, step=4*days)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True, shuffle=False, prefetch_factor=2,persistent_workers=False)
    if rank == 0:
        logger.info(f"Dataset loaded successfully; loader length: {len(train_loader)}")
        

    step_start_time = time.time()
    model.train() 
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
            
            
            for a in range(days):
                
                if a == 0:
                    input = data.to(device, non_blocking=True)
                else:
                    input = input.detach().to(device, non_blocking=True)
                for k in range(4*a, 4*(a+1)):
                    target[k] = target[k].to(device, non_blocking=True)


                with autocast(enabled=True, dtype=torch.float16): 
                    total_loss = 0.0
                    total_MAE = 0.0
                    for j in range(4*a, 4*(a+1)):
                        output_norm = model(input) 
                        loss_rmse = loss_fn_rmse(output_norm, target[j]) 
                        loss = loss_rmse 
                        total_loss += loss
                        total_MAE += loss_rmse.item()
                        
                        input_1 = output_norm.unsqueeze(1) # (B, 1, 69, H, W)
                        input_0 = input[:,1,:,:,:].unsqueeze(1) # (B, 1, 69, H, W)
                        input = torch.cat([input_0, input_1], dim=1) # (B, 2, 69, H, W)


                    loss = total_loss / 4
                    loss_MAE = total_MAE / 4

                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f" Batch {i+1}: Loss is NaN or Inf! Skipping batch.")
                    
                    continue 

                
                scaler.scale(loss).backward() 
                
                scaler.step(optimizer) 
                
                scaler.update() 
                
                optimizer.zero_grad()

                if i % log_interval == 0 and rank == 0:
                    end_time = time.time()
                    logger.info(f"----------------------------------train_step {train_step} finished; elapsed time: {end_time - start_time:.2f}s; day {a+1} average MAE: {loss_MAE:.6f}; loss_dtype: {loss.dtype}; lr: {optimizer.param_groups[0]['lr']:.8f}----------------------------------")
                
            
            del data, target
            if 'output_norm' in locals(): del output_norm
            del loss, loss_MAE, input, input_0, input_1
            torch.cuda.empty_cache()
            gc.collect()

            if train_step % 100 == 0 :
                
                if rank == 0:
                    logger.info(f"train_step {train_step}: saving checkpoint...")
                    ckpt_path = os.path.join(baseline_path, saved_model_dir, f"model_step_{train_step}.pt")
                    save_checkpoint(
                        path=ckpt_path,
                        model=model,
                        optimizer=None,
                        scheduler=None,  
                        epoch=None                  
                    )
                    logger.info(f"step {train_step}: saved to {ckpt_path}")
                dist.barrier()

    
    step_end_time = time.time()
    step_duration = step_end_time - step_start_time


    
    if rank == 0:
        current_lr = optimizer.param_groups[0]['lr'] 
        logger.info(f'Train completed in {step_duration/3600:.2f}h. LR: {current_lr:.8f}')

    if rank == 0:
        logger.info("Training finished.")

    cleanup_distributed()

if __name__ == "__main__":
    
    train()
