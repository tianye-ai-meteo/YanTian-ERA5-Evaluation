import os

def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")

'数据集配置'
train_start_year = 2010
train_end_year = 2019
val_start_year = 2019
val_end_year = 2019
'训练配置'
# 如果从0开始训练，设置continue_train为False，不需要关注model_epoch、continue_train_epoch；
# 如果训练过程中有训练中断，重新训练时需要加载中间保存的模型参数，则设置continue_train为True，并设置model_epoch、continue_train_epoch
continue_train = _env_bool("YANTIAN_FINETUNE_CONTINUE_TRAIN", True)
model_epoch = 36 # 如果训练过程中有训练中断，重新训练时需要加载中间保存的模型参数，此参数为加载之前在某个epoch保存的模型的epoch数
model_step = 4

# 优化器
use_new_lr = True # 是否使用新的学习率
new_lr = 1e-7 # 新的学习率
new_beta = (0.9, 0.95) # 新的beta
new_weight_decay = 0.1 # 新的weight_decay

# 迭代步数
days = 15 # 学习天气过程演变天数
train_steps = 3000


'分布式训练配置'
train_world_size = int(os.environ.get("YANTIAN_TRAIN_WORLD_SIZE", 8)) # 使用多少个GPU
save_epoch_interval = 1 # 每训练多少个epoch保存一次模型

'数据加载与batch size配置'
train_total_batch_size = int(os.environ.get("YANTIAN_TRAIN_TOTAL_BATCH_SIZE", 8)) # 所有GPU总计的batch size
accumulation_steps = int(os.environ.get("YANTIAN_ACCUMULATION_STEPS", 1)) # 累积步数（模拟 batch_size=4）
batch_size = train_total_batch_size // train_world_size # 每个GPU的batch size
num_workers = int(os.environ.get("YANTIAN_NUM_WORKERS", 4)) # 数据加载的线程数

'使用数据集文件夹'
data_dir = os.environ.get("YANTIAN_TRAIN_DATA_DIR", "/g18831412218lty/ERA5-Global-LM-1-norm-fp32")

'结果保存配置'
saved_model_dir = os.environ.get("YANTIAN_FINETUNE_CHECKPOINT_DIR", "model_saved/") # 训练过程中保存的模型文件夹
saved_picture_path = os.environ.get("YANTIAN_FINETUNE_FIGURE_DIR", "rmse_picture/") # 训练过程中保存的图片文件夹

'日志保存配置'
log_path = os.environ.get("YANTIAN_FINETUNE_LOG_PATH", "train.log") # 训练过程中保存的日志文件夹
log_interval = int(os.environ.get("YANTIAN_LOG_INTERVAL", 1)) # 每训练多少个batch保存一次日志
