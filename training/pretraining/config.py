import os

def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")

# Dataset configuration
train_start_year = 1987
train_end_year = 2016
val_year = 2017

# Training configuration

# Resume-training configuration


continue_train = _env_bool('YANTIAN_PRETRAIN_CONTINUE_TRAIN', False)
model_epoch = 20 
continue_train_epoch = 20 

use_new_lr = False  
new_lr = 2e-4 
new_beta = (0.9, 0.95) 
new_weight_decay = 0.1 

new_lr_min = 1e-5

train_epoch_losses = []
val_epoch_losses = []

# Initial-training configuration
# Learning-rate configuration
epochs = 50 
lr = 2e-4
lr_min = 1e-7
save_epoch_interval = 10 

# Data loading and batch-size configuration
data_dir = os.environ.get('YANTIAN_TRAIN_DATA_DIR', '/g18831412218lty/ERA5-Global-LM-1-norm-fp32')
train_world_size = int(os.environ.get('YANTIAN_TRAIN_WORLD_SIZE', 8)) 
train_total_batch_size = int(os.environ.get('YANTIAN_TRAIN_TOTAL_BATCH_SIZE', 32)) 
accumulation_steps = int(os.environ.get('YANTIAN_ACCUMULATION_STEPS', 1)) 
batch_size = train_total_batch_size // train_world_size 
num_workers = int(os.environ.get('YANTIAN_NUM_WORKERS', 4)) 

# Output configuration
saved_model_dir = os.environ.get('YANTIAN_PRETRAIN_CHECKPOINT_DIR', 'model_saved/') 
saved_picture_path = os.environ.get('YANTIAN_PRETRAIN_FIGURE_DIR', 'rmse_picture/') 

# Logging configuration
log_path = os.environ.get('YANTIAN_PRETRAIN_LOG_PATH', 'train.log') 
log_interval = int(os.environ.get('YANTIAN_LOG_INTERVAL', 10)) 
