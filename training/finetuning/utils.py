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

    
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_path)
    
    # with open(log_file_path, 'w') as f:
    #     pass
    file_handler = logging.FileHandler(log_file_path)



    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') 
    file_handler.setFormatter(formatter)


    
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger

def setup_distributed(backend='nccl'):
    """Initialize torch.distributed from environment variables."""
    
    if dist.is_initialized():
        return

    
    
    
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    print(f"Initializing distributed training environment (Process {rank}/{world_size}, Local Rank {local_rank})")

    
    
    torch.cuda.set_device(local_rank)
    print(f"Process {rank}: setting CUDA device to {local_rank} ({torch.cuda.get_device_name(local_rank)})")

    
    dist.init_process_group(backend=backend, init_method='env://',
                              world_size=world_size, rank=rank)

    print(f"Process {rank}: distributed training environment initialized successfully.")
    
    dist.barrier()

def cleanup_distributed():
    """Tear down the distributed process group if it is active."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed training environment destroyed.")


def unnormalize_layer_torch(data, avg_tensor, std_tensor):
    """Convert normalized tensors back to physical units."""
    
    # avg_tensor = avg_tensor.to(data.device)
    # std_tensor = std_tensor.to(data.device)

    
    
    unnormalized_data = data * std_tensor + avg_tensor
    return unnormalized_data

def get_avg_std(path):
    """Load channel-wise mean and standard-deviation arrays from JSON."""
    
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
    
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()

def load_model_state_dict(model, state_dict, strict=True):
    
    target = model.module if hasattr(model, "module") else model
    missing, unexpected = target.load_state_dict(state_dict, strict=strict)
    return missing, unexpected

def save_checkpoint(path, model, optimizer, scheduler, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": get_model_state_dict(model),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,  
    }
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu", strict=True):
    ckpt = torch.load(path, map_location=map_location)
    missing, unexpected = load_model_state_dict(model, ckpt["model"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1  
    return start_epoch, missing, unexpected
