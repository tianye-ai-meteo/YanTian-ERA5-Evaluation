import os

def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")

# Dataset configuration
train_start_year = 2010
train_end_year = 2019
val_start_year = 2019
val_end_year = 2019
# Training configuration


continue_train = _env_bool("YANTIAN_FINETUNE_CONTINUE_TRAIN", True)
model_epoch = 36 
model_step = 4


use_new_lr = True 
new_lr = 1e-7 
new_beta = (0.9, 0.95) 
new_weight_decay = 0.1 


days = 15 
train_steps = 3000


# Training runtime configuration
train_world_size = int(os.environ.get("YANTIAN_TRAIN_WORLD_SIZE", 8)) 
save_epoch_interval = 1 

# Data loading and batch-size configuration
train_total_batch_size = int(os.environ.get("YANTIAN_TRAIN_TOTAL_BATCH_SIZE", 8)) 
accumulation_steps = int(os.environ.get("YANTIAN_ACCUMULATION_STEPS", 1)) 
batch_size = train_total_batch_size // train_world_size 
num_workers = int(os.environ.get("YANTIAN_NUM_WORKERS", 4)) 

# Training data directory
data_dir = os.environ.get("YANTIAN_TRAIN_DATA_DIR", "/g18831412218lty/ERA5-Global-LM-1-norm-fp32")

# Output configuration
saved_model_dir = os.environ.get("YANTIAN_FINETUNE_CHECKPOINT_DIR", "model_saved/") 
saved_picture_path = os.environ.get("YANTIAN_FINETUNE_FIGURE_DIR", "rmse_picture/") 

# Logging configuration
log_path = os.environ.get("YANTIAN_FINETUNE_LOG_PATH", "train.log") 
log_interval = int(os.environ.get("YANTIAN_LOG_INTERVAL", 1)) 
