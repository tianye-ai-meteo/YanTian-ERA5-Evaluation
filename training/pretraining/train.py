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
from dataset import BaselineDataset, BaselineValDataset
from config import *
import torch.distributed as dist
from loss import LatitudeWeightedMAELoss, DynamicWeightedMAELoss
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import setup_logger, setup_distributed, cleanup_distributed, save_checkpoint, load_checkpoint, count_parameters
from torch.cuda.amp import autocast, GradScaler
from timm.models.layers import DropPath
from torch.optim.lr_scheduler import CosineAnnealingLR

# Example launch command:
# nohup torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29501 train.py >nohup_train.log 2>&1 &
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

    
    train_dataset = BaselineDataset(start_year=train_start_year, end_year=train_end_year)
    val_dataset = BaselineValDataset(start_year=val_year, end_year=val_year)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True, shuffle=False, prefetch_factor=5,persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True, drop_last=True, shuffle=False, prefetch_factor=5,persistent_workers=True)
    if rank == 0:
        logger.info(f"Dataset loaded successfully; loader length: {len(train_loader)}")

    
    num_train_steps_per_epoch = len(train_loader)

    
    model = BaselineModel().to(device)
    
    if rank == 0:
        logger.info(f"Model initialized successfully on {device} (FP32).")


    
    
    loss_mae = DynamicWeightedMAELoss().to(device) 
    if rank == 0:
        logger.info(f"MAE loss function created successfully on {device} (FP32).")

    
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    if not use_new_lr and rank == 0:
        logger.info(f"Optimizer created successfully on {device}; LR: {lr}; training steps: {num_train_steps_per_epoch*epochs}.")

    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=num_train_steps_per_epoch*epochs, 
        eta_min=lr_min      
    )
    
    scaler = GradScaler(enabled=True) 
    if rank == 0:
        logger.info(f"GradScaler created successfully on {device} for mixed-precision training.")


    start_epoch = 0 

    
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    
    if dist.get_rank() == 0:
        num_params = count_parameters(model.module)  
        logger.info(f"Model parameters: {num_params:,}")

    if continue_train:
        next_epoch, missing, unexpected = load_checkpoint(
            os.path.join(baseline_path, saved_model_dir, f'model_epoch_{model_epoch}.pt'),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler_cosine,
            map_location=device,   
            strict=True        
        )
        start_epoch = next_epoch
        if missing or unexpected:
            logger.warning(f"Missing parameters while loading checkpoint: {missing}")
            logger.warning(f"Unexpected parameters while loading checkpoint: {unexpected}")
        else:
            logger.info(f"Checkpoint loaded successfully; resuming from epoch {start_epoch}.")

        if rank == 0:
                logger.info(f"Learning-rate scheduler loaded successfully. LR: {optimizer.param_groups[0]['lr']:.8f}")

        if use_new_lr:
            optimizer = optim.AdamW(model.parameters(), lr=new_lr, betas=new_beta, weight_decay=new_weight_decay)
            scheduler_cosine = CosineAnnealingLR(
                optimizer,
                T_max=num_train_steps_per_epoch*continue_train_epoch, 
                eta_min=new_lr_min      
            )
            if rank == 0:
                logger.info(f"Using a new learning-rate scheduler. New LR: {new_lr}, new min LR: {new_lr_min}, new beta: {new_beta}, new weight_decay: {new_weight_decay}, continued training steps: {num_train_steps_per_epoch*continue_train_epoch}")

        dist.barrier()

    
    if rank == 0:
        os.makedirs(os.path.join(baseline_path, saved_model_dir), exist_ok=True)
        os.makedirs(os.path.join(baseline_path, saved_picture_path), exist_ok=True)

    
    total_epochs = continue_train_epoch if continue_train else epochs
    end_epoch = start_epoch + total_epochs

    if rank == 0:
        logger.info(f"Starting training from epoch {start_epoch} to {end_epoch - 1}.")

    
    for epoch in range(start_epoch, end_epoch):
        
        start_drop_path = 0.0
        end_drop_path = 0.2

        total_epochs = end_epoch - start_epoch
        current_epoch = epoch - start_epoch
        find_drop = False
        for i, module in enumerate(model.modules()):
            if isinstance(module, DropPath):
                module.drop_prob = start_drop_path + (end_drop_path - start_drop_path) * current_epoch / total_epochs
                if rank == 0 and not find_drop :
                    logger.info(f"------- epoch: {epoch} ---------")
                    logger.info(f"Drop rate: {module.drop_prob}")
                    logger.info(f"----------------------------------")
                find_drop = True
            
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        model.train() 
        epoch_train_loss = 0.0
        # all_l_aux = 0.0
        batch_count = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):

            if rank == 0 and i == 0:
                start_time = time.time()

            data, target = batch
            
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            
            with autocast(enabled=True, dtype=torch.float16):  
                output_norm = model(data) 
                loss_MAE = loss_mae(output_norm, target) 
                loss = loss_MAE 
                loss = loss / accumulation_steps

            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Epoch {epoch}, Batch {i+1}: Loss is NaN or Inf! Skipping batch.")
                continue 

            
            scaler.scale(loss).backward() 

            if (i+1) % accumulation_steps == 0:
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer) 
                
                scaler.update() 
                 
                optimizer.zero_grad()

            epoch_train_loss += loss_MAE.item() 
            # all_l_aux += l_aux.item()
            batch_count += 1

            
            scheduler_cosine.step()
            # ------------------------------

            if i % log_interval == 0 and rank == 0:
                end_time = time.time()
                logger.info(f"----------------------------------batch {i} finished; elapsed time: {end_time - start_time:.2f}s; loss_MAE: {loss_MAE:.6f}; loss_dtype: {loss.dtype}; lr: {optimizer.param_groups[0]['lr']:.8f}----------------------------------")


        
        avg_train_loss = epoch_train_loss / batch_count if batch_count > 0 else 0
        train_epoch_losses.append(avg_train_loss)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        
        model.eval() 
        epoch_val_loss = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            with autocast(enabled=True, dtype=torch.float16): 
                for i, val_batch in enumerate(val_loader):
                    if i%10 != 0 :
                        continue
                    if val_batch is None or val_batch[0] is None or val_batch[1] is None:
                        logger.warning(f"Epoch {epoch}: Skipped None validation batch.")
                        continue

                    val_data, val_target = val_batch
                    
                    val_data = val_data.to(device, non_blocking=True)
                    val_target = val_target.to(device, non_blocking=True)
                    # ------------------------------------------

                    
                    val_output_norm = model(val_data) 
                    
                    val_loss_MAE = loss_mae(val_output_norm, val_target) 
                    val_loss = val_loss_MAE
                    # ------------------------------------

                    
                    if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                        epoch_val_loss += val_loss.item() 
                        val_batch_count += 1
                    else:
                        logger.warning(f"Epoch {epoch}: Validation loss is NaN/Inf in a batch.")

        
        avg_val_loss = epoch_val_loss / val_batch_count if val_batch_count > 0 else 0
        val_epoch_losses.append(avg_val_loss)

        
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr'] 
            logger.info(f'Epoch [{epoch}/{end_epoch-1}]  completed in {epoch_duration/3600:.2f}h. Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.8f}')

        
        if epoch % save_epoch_interval == 0 or epoch == end_epoch - 1: 
            if rank == 0:
                logger.info(f"Epoch {epoch}: saving checkpoint...")
                ckpt_path = os.path.join(baseline_path, saved_model_dir, f"model_epoch_{epoch}.pt")
                save_checkpoint(
                    path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler_cosine,  
                    epoch=epoch                  
                )
                logger.info(f"Epoch {epoch}: saved to {ckpt_path}")
            dist.barrier()  


        
        if rank == 0:
            try:
                plt.figure(figsize=(10, 6))
                
                epochs_range = range(len(train_epoch_losses))
                plt.plot(epochs_range, train_epoch_losses, label='Train Loss', color='red', linestyle='-')
                plt.plot(epochs_range, val_epoch_losses, label='Val Loss', color='blue', linestyle='--')
                plt.xlabel('Epoch')
                plt.ylabel('Average Loss')
                plt.title('Training and Validation Loss Curve')
                plt.legend()
                plt.grid(True)
                
                loss_curve_filename = f'loss_curve_6h{"_continue" if continue_train else ""}.png'
                loss_curve_path = os.path.join(baseline_path, saved_picture_path, loss_curve_filename)
                plt.savefig(loss_curve_path)
                plt.close() 
                # logger.info(f"Loss curve saved to {loss_curve_path}")
            except Exception as e:
                logger.error(f"Failed to plot or save loss curve at epoch {epoch}: {e}")

    if rank == 0:
        logger.info("Training finished.")


    cleanup_distributed()

if __name__ == "__main__":
    
    train()
