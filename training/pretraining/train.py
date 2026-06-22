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

'''
nohup torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29501 train.py >nohup_train.log 2>&1 &
'''
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

    # 准备数据集
    train_dataset = BaselineDataset(start_year=train_start_year, end_year=train_end_year)
    val_dataset = BaselineValDataset(start_year=val_year, end_year=val_year)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    # shuffle=False 对于 sampler 后的 DataLoader 是推荐的，因为 sampler 已经处理了 shuffle
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True, shuffle=False, prefetch_factor=5,persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True, drop_last=True, shuffle=False, prefetch_factor=5,persistent_workers=True)
    if rank == 0:
        logger.info(f"数据集加载成功, 数据集大小: {len(train_loader)}")

    # 获取每个 epoch 的训练步数，用于计算 warmup 的总步数
    num_train_steps_per_epoch = len(train_loader)

    # 准备模型
    model = BaselineModel().to(device)
    # 模型保持默认的 FP32 精度
    if rank == 0:
        logger.info(f"{device}上模型初始化成功 (FP32)。")


    # --- 定义损失函数 ---
    # 损失函数通常不需要手动转换类型，它们会根据输入张量的类型进行计算
    loss_mae = DynamicWeightedMAELoss().to(device) # 移动到设备
    if rank == 0:
        logger.info(f"{device}上MAE损失函数创建成功 (FP32).")

    # --- 定义优化器 ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    if not use_new_lr and rank == 0:
        logger.info(f"{device}上优化器创建成功，LR: {lr}，训练步数: {num_train_steps_per_epoch*epochs}。")

    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=num_train_steps_per_epoch*epochs, # 余弦衰减的总 step 数 (按用户指定)
        eta_min=lr_min      # 衰减到的最小学习率
    )
    # --- 初始化 GradScaler ---
    scaler = GradScaler(enabled=True) # enabled=True 表示启用混合精度 (FP16)
    if rank == 0:
        logger.info(f"{device}上GradScaler创建成功，用于FP16混合精度训练。")


    start_epoch = 0 # 默认从 epoch 0 开始

    # DDP 封装模型
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # 打印模型参数量
    if dist.get_rank() == 0:
        num_params = count_parameters(model.module)  # 注意这里要用 model.module
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
            logger.warning(f"加载检查点时缺少参数: {missing}")
            logger.warning(f"加载检查点时发现意外参数: {unexpected}")
        else:
            logger.info(f"加载检查点成功，从 epoch {start_epoch} 继续训练。")

        if rank == 0:
                logger.info(f"学习率调度器加载成功。学习率: {optimizer.param_groups[0]['lr']:.8f}")

        if use_new_lr:
            optimizer = optim.AdamW(model.parameters(), lr=new_lr, betas=new_beta, weight_decay=new_weight_decay)
            scheduler_cosine = CosineAnnealingLR(
                optimizer,
                T_max=num_train_steps_per_epoch*continue_train_epoch, # 余弦衰减的总 step 数 (按用户指定)
                eta_min=new_lr_min      # 衰减到的最小学习率
            )
            if rank == 0:
                logger.info(f"使用新的学习率调度器。新学习率: {new_lr}, 新学习率最小值: {new_lr_min}, 新beta: {new_beta}, 新weight_decay: {new_weight_decay}, 继续训练步数: {num_train_steps_per_epoch*continue_train_epoch}")

        dist.barrier()

    # 创建保存模型的目录
    if rank == 0:
        os.makedirs(os.path.join(baseline_path, saved_model_dir), exist_ok=True)
        os.makedirs(os.path.join(baseline_path, saved_picture_path), exist_ok=True)

    # 确定总训练 epochs
    total_epochs = continue_train_epoch if continue_train else epochs
    end_epoch = start_epoch + total_epochs

    if rank == 0:
        logger.info(f"Starting training from epoch {start_epoch} to {end_epoch - 1}.")

    # 训练循环
    for epoch in range(start_epoch, end_epoch):
        # 1. 调度 DropPath 率: 从0.1 增加到0.2
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
                    logger.info(f"Drop 率: {module.drop_prob}")
                    logger.info(f"----------------------------------")
                find_drop = True
            
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        model.train() # 设置为训练模式
        epoch_train_loss = 0.0
        # all_l_aux = 0.0
        batch_count = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):

            if rank == 0 and i == 0:
                start_time = time.time()

            data, target = batch
            # 数据移动到设备时，保持默认的 FP32 精度
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # 使用 FP16 混合精度进行前向传播和损失计算
            with autocast(enabled=True, dtype=torch.float16):  # 启用 autocast 上下文，使用 FP16
                output_norm = model(data) # 模型输出在 autocast 下可能是 FP16
                loss_MAE = loss_mae(output_norm, target) # 损失计算也在 autocast 上下文中
                loss = loss_MAE 
                loss = loss / accumulation_steps

            # 检查 loss 是否为 NaN 或 Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Epoch {epoch}, Batch {i+1}: Loss is NaN or Inf! Skipping batch.")
                continue # 跳过这个 batch

            # --- 反向传播 (使用 scaler.scale) ---
            scaler.scale(loss).backward() # 使用 GradScaler 缩放损失并执行反向传播

            if (i+1) % accumulation_steps == 0:
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # --- 更新权重 (使用 scaler.step) ---
                scaler.step(optimizer) # GradScaler 负责 unscale 梯度并调用 optimizer.step()
                # --- 更新 scaler 的缩放因子 ---
                scaler.update() # 更新 GradScaler 的缩放因子
                 # 梯度清零
                optimizer.zero_grad()

            epoch_train_loss += loss_MAE.item() # loss.item() 会将张量转为 float
            # all_l_aux += l_aux.item()
            batch_count += 1

            # --- 学习率更新 (统一按步进行) ---
            scheduler_cosine.step()
            # ------------------------------

            if i % log_interval == 0 and rank == 0:
                end_time = time.time()
                logger.info(f"----------------------------------batch {i} 完成，用时: {end_time - start_time:.2f}s, loss_MAE: {loss_MAE:.6f}, loss_dtype: {loss.dtype}, lr: {optimizer.param_groups[0]['lr']:.8f}----------------------------------")


        # 计算并记录 epoch 平均训练损失
        avg_train_loss = epoch_train_loss / batch_count if batch_count > 0 else 0
        train_epoch_losses.append(avg_train_loss)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # --- 验证过程 ---
        model.eval() # 设置为评估模式
        epoch_val_loss = 0.0
        val_batch_count = 0
        # 使用 no_grad 进行验证，需要 autocast
        with torch.no_grad():
            with autocast(enabled=True, dtype=torch.float16): # <--- 在验证时也使用 autocast 和 FP16
                for i, val_batch in enumerate(val_loader):
                    if i%10 != 0 :
                        continue
                    if val_batch is None or val_batch[0] is None or val_batch[1] is None:
                        logger.warning(f"Epoch {epoch}: Skipped None validation batch.")
                        continue

                    val_data, val_target = val_batch
                    # --- 将验证数据移动到设备，保持默认 FP32 精度 ---
                    val_data = val_data.to(device, non_blocking=True)
                    val_target = val_target.to(device, non_blocking=True)
                    # ------------------------------------------

                    # 模型前向传播
                    val_output_norm = model(val_data) # 输出在 autocast 下可能是 FP16
                    # 计算损失
                    val_loss_MAE = loss_mae(val_output_norm, val_target) # 计算也在 autocast 上下文中
                    val_loss = val_loss_MAE
                    # ------------------------------------

                    # 检查验证损失是否 NaN/Inf
                    if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                        epoch_val_loss += val_loss.item() # val_loss.item() 转为 float
                        val_batch_count += 1
                    else:
                        logger.warning(f"Epoch {epoch}: Validation loss is NaN/Inf in a batch.")

        # 计算平均验证损失
        avg_val_loss = epoch_val_loss / val_batch_count if val_batch_count > 0 else 0
        val_epoch_losses.append(avg_val_loss)

        # 记录日志，标明使用的损失类型
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr'] # 获取当前学习率
            logger.info(f'Epoch [{epoch}/{end_epoch-1}]  completed in {epoch_duration/3600:.2f}h. Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.8f}')

        # 保存 checkpoint 
        if epoch % save_epoch_interval == 0 or epoch == end_epoch - 1: # 每隔 N 个 epoch 或最后一个 epoch 保存
            if rank == 0:
                logger.info(f"Epoch {epoch}: 保存 checkpoint...")
                ckpt_path = os.path.join(baseline_path, saved_model_dir, f"model_epoch_{epoch}.pt")
                save_checkpoint(
                    path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler_cosine,  # 保存调度器状态
                    epoch=epoch                  # 保存“已完成”的 epoch
                )
                logger.info(f"Epoch {epoch}: 已保存到 {ckpt_path}")
            dist.barrier()  


        # 绘制并保存 loss 曲线图
        if rank == 0:
            try:
                plt.figure(figsize=(10, 6))
                # 绘制从开始到当前 epoch 的所有 loss
                epochs_range = range(len(train_epoch_losses))
                plt.plot(epochs_range, train_epoch_losses, label='Train Loss', color='red', linestyle='-')
                plt.plot(epochs_range, val_epoch_losses, label='Val Loss', color='blue', linestyle='--')
                plt.xlabel('Epoch')
                plt.ylabel('Average Loss')
                plt.title('Training and Validation Loss Curve')
                plt.legend()
                plt.grid(True)
                # 根据是否继续训练选择不同的文件名
                loss_curve_filename = f'loss_curve_6h{"_continue" if continue_train else ""}.png'
                loss_curve_path = os.path.join(baseline_path, saved_picture_path, loss_curve_filename)
                plt.savefig(loss_curve_path)
                plt.close() # 关闭图形，释放内存
                # logger.info(f"Loss curve saved to {loss_curve_path}")
            except Exception as e:
                logger.error(f"Failed to plot or save loss curve at epoch {epoch}: {e}")

    if rank == 0:
        logger.info("Training finished.")


    cleanup_distributed()

if __name__ == "__main__":
    # 可以在这里添加命令行参数解析来覆盖 config.py 中的设置
    train()