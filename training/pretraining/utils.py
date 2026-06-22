import logging
import os
import sys
import torch
import torch.distributed as dist
from config import log_path
from torch.nn.parallel import DistributedDataParallel as DDP
import json

def setup_logger():
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    # 文件处理器
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_path)
    # 清空日志文件内容（如果需要每次重新开始记录）
    # with open(log_file_path, 'w') as f:
    #     pass
    file_handler = logging.FileHandler(log_file_path)



    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # 添加 levelname
    file_handler.setFormatter(formatter)


    # 防止重复添加 handler
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger

def setup_distributed(backend='nccl'):
    """
    1、需要将分布式环境设置放在训练函数的第一顺序
    2、会读取静态环境变量如'WORLD_SIZE'、'RANK'、'LOCAL_RANK'。（通过torchrun启动函数自动识别设置）
    3、会为当前线程绑定cuda_devide
    4、为所有线程设置后段通讯
    """
    # 检查分布式训练是否已经初始化（即检查是否已经执行dist.init_process_group）
    if dist.is_initialized():
        return

    # WORLD_SIZE 总进程（所有节点的GPUS之和）
    # RANK 所有节点的所有GPUS的统计编号（0-WORLD_SIZE-1）.
    # LOCAL_RANK 单节点的GPUS编号 (0 to GPUs_per_node-1).
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    print(f"初始化分布式训练环境 (Process {rank}/{world_size}, Local Rank {local_rank})")

    # --- 设置CUDA设备 ---
    # 每个进程必须操作其分配的GPU。DDP依赖于此绑定。Tutel的MoE通信也假设进程正确绑定到设备。
    torch.cuda.set_device(local_rank)
    print(f"Process {rank}: 设置CUDA设备为 {local_rank} ({torch.cuda.get_device_name(local_rank)})")

    # --- 初始化分布式训练环境 ---
    dist.init_process_group(backend=backend, init_method='env://',
                              world_size=world_size, rank=rank)

    print(f"Process {rank}: 分布式训练环境初始化成功。")
    # Barrier确保所有进程都到达此点后再继续。
    dist.barrier()

def cleanup_distributed():
    """ 销毁分布式训练环境 """
    if dist.is_initialized():
        dist.destroy_process_group()
        print("分布式训练环境销毁。")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unnormalize_layer_torch(data, avg_tensor, std_tensor):
    """
    使用 PyTorch 张量操作将数据反归一化。
    输入:
        data (torch.Tensor): 归一化后的数据张量 (B, C, H, W)。
        avg_tensor (torch.Tensor): 形状为 (1, C, 1, 1) 的均值张量。
        std_tensor (torch.Tensor): 形状为 (1, C, 1, 1) 的标准差张量。
    输出:
        torch.Tensor: 反归一化后的数据张量，与输入 data 具有相同的设备，
                      数据类型通常会提升为 avg/std 张量的数据类型 (例如 float32)。
    """
    # 确保张量在同一设备 (虽然在 train 函数中已经保证)
    # avg_tensor = avg_tensor.to(data.device)
    # std_tensor = std_tensor.to(data.device)

    # 执行反归一化: data * std + avg
    # PyTorch 会自动处理广播机制
    unnormalized_data = data * std_tensor + avg_tensor
    return unnormalized_data

def get_avg_std(path):
    """
    获取数据的平均值和标准差用于归一化
    
    参数:
        path: 统计数据的JSON文件路径
    
    返回:
        pressure_avg: 气压层数据的平均值
        pressure_std: 气压层数据的标准差
        avg_sur: 地表数据的平均值
        std_sur: 地表数据的标准差
    """
    # 加载年平均值和标准差
    with open(path, 'r') as file:
        json_data = json.load(file)
    avg_list = json_data['avg']
    std_list = json_data['std']
    pressure_avg = avg_list[7:]
    pressure_std = std_list[7:]
    surface_avg = avg_list[:6]
    surface_std = std_list[:6]
    return pressure_avg, pressure_std, surface_avg, surface_std

def get_model_state_dict(model):
    # 兼容 DDP / 非DDP
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()

def load_model_state_dict(model, state_dict, strict=True):
    # 兼容 DDP / 非DDP
    target = model.module if hasattr(model, "module") else model
    missing, unexpected = target.load_state_dict(state_dict, strict=strict)
    return missing, unexpected

def save_checkpoint(path, model, optimizer, scheduler, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": get_model_state_dict(model),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,  # 保存“已完成”的 epoch
    }
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu", strict=True):
    ckpt = torch.load(path, map_location=map_location)
    missing, unexpected = load_model_state_dict(model, ckpt["model"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1  # 下一个要训练的 epoch
    return start_epoch, missing, unexpected