import os
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import logging
"""
Author: lity
Date: 2025-05-21
Description: 
    文件功能描述: 将nc数据读取，归一化，并保存npy格式
    文件启动方式描述: nohup python process_data_to_npy.py &
    文件函数主要功能描述: 
        1、加载均值和标准差数据,并排列为nc读取数据并合并之后的顺序,并扩展为相同的维度数量（dim，dim，dim）
        2、根据年份生成nc文件列表
        3、加载指定时间的数据 (从 .nc 文件)，返回(84, 721, 1440) 的 torch.float32 张量
        4、将数据归一化，并保存为npy格式
"""
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ERA5_NC_ROOT = os.environ.get("YANTIAN_ERA5_NC_ROOT", "/home/dataset/ERA5-Global-LM")
ERA5_NORM_OUTPUT_ROOT = os.environ.get("YANTIAN_ERA5_NORM_OUTPUT_ROOT", "/home/dataset/ERA5-Global-LM-1-norm")

def setup_logger(start_year, end_year):
    logger = logging.getLogger('data_process_1_norm'+str(start_year)+str(end_year))
    logger.setLevel(logging.INFO)

    # 文件处理器
    log_file_path = os.path.join(SCRIPT_DIR, 'data_process_1_norm'+str(start_year)+str(end_year)+'.log')
    file_handler = logging.FileHandler(log_file_path)

    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # 添加 levelname
    file_handler.setFormatter(formatter)


    # 防止重复添加 handler
    if not logger.handlers:
        logger.addHandler(file_handler)


    return logger

def load_avg_std():
    """加载均值和标准差数据,并排列为nc读取数据并合并之后的顺序,并扩展为相同的维度数量（dim，dim，dim）：
    0-12: z[1-13],50hpa-1000hpa
    13-25: r[1-13],50hpa-1000hpa
    26-38: t[1-13],50hpa-1000hpa
    39-51: u[1-13],50hpa-1000hpa
    52-64: v[1-13],50hpa-1000hpa
    65-77: w[1-13],50hpa-1000hpa
    78-83: u10,v10,d2m,t2m,msl,sp
    """
    avg_std_path = os.path.join(SCRIPT_DIR, 'statistics.json')
    with open(avg_std_path, 'r') as file:
        json_data = json.load(file)
    avg_list = json_data['avg']
    std_list = json_data['std']
    pressure_avg = np.array(avg_list[7:-13])
    pressure_std = np.array(std_list[7:-13])
    surface_avg_1 = avg_list[0:2]
    surface_avg_2 = avg_list[3:5]
    surface_avg = np.concatenate([surface_avg_1, surface_avg_2], axis=0)
    surface_std_1 = std_list[0:2]
    surface_std_2 = std_list[3:5]
    surface_std = np.concatenate([surface_std_1, surface_std_2], axis=0)
    avg = np.concatenate([pressure_avg, surface_avg], axis=0)
    std = np.concatenate([pressure_std, surface_std], axis=0)
    # np.expand_dims 不会改变数组的数据类型
    avg = np.expand_dims(avg, axis=(1,2))
    std = np.expand_dims(std, axis=(1,2))

    return avg, std
        




def get_file_path(dt, file_type):
    """获取指定时间和类型的文件路径 (nc格式)
    Args:
        dt (datetime): 时间点
        file_type (str): 文件类型，'pressure'或'single'
    Returns:
        str: 文件完整路径
    """
    year_dir = os.path.join(ERA5_NC_ROOT, str(dt.year))
    date_dir = os.path.join(year_dir, dt.strftime('%Y%m%d'))
    if file_type == 'pressure':
        # 更新文件名后缀为 .nc
        filename = f'ERA5_Global_LM_Pressure_{dt.strftime("%Y%m%d%H")}.nc' # <--- 修改后缀
    else:  # single
        # 更新文件名后缀为 .nc
        filename = f'ERA5_Global_LM_Single_{dt.strftime("%Y%m%d%H")}.nc' # <--- 修改后缀
    return os.path.join(date_dir, filename)

def _generate_time_indices(start_year, end_year):
    """生成时间索引列表
    每天的采样时刻为00、06、12、18
    预测模式为：
    1. 18,00 -> 06 (对起始年份第一天不可用)
    2. 00,06 -> 12
    3. 06,12 -> 18
    4. 12,18 -> 次日00
    """
    time_indices = []
    
    for year in range(start_year, end_year + 1):
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 18)
        # 处理正常情况
        current_date = start_date
        while current_date <= end_date:
            time_indices.append(current_date)
            current_date += timedelta(hours=6)
    
    return sorted(time_indices)  # 确保时间顺序

def _load_data_for_time(dt, avg, std, logger):
    """加载指定时间的数据 (从 .nc 文件)，返回(84, 721, 1440) 的 float32 张量
        1、加载pressure层数据和single层数据，并合并为(84, 721, 1440)
            0-12: z[1-13],50hpa-1000hpa
            13-25: r[1-13],50hpa-1000hpa
            26-38: t[1-13],50hpa-1000hpa
            39-51: u[1-13],50hpa-1000hpa
            52-64: v[1-13],50hpa-1000hpa
            65-68: u10,v10,t2m,msl
        2、使用归一化数据进行整体归一化
        3、转化为float32
        4、保存为npy格式
    """
    upper_vars = ['z', 'r', 't', 'u', 'v']
    surface_vars = ['u10', 'v10', 't2m', 'msl']
    try:
        # 加载pressure层数据
        pressure_file = get_file_path(dt, 'pressure')
        upper_data_list = []
        with xr.open_dataset(pressure_file) as ds:
            for var in upper_vars:
                # 读取数据并确保转换为 float32
                data = ds[var][:].astype(np.float32) # 形状: (1, 13, 721, 1440)
                downsampled = data.coarsen(latitude=4, longitude=4, boundary="trim").mean()
                upper_data_list.append(downsampled)
        # 拼接高空数据，结果为 float32
        upper_data = np.concatenate(upper_data_list, axis=1).squeeze(0) # 形状: (65,180,360), 类型: float32


        # 加载single层数据
        single_file = get_file_path(dt, 'single')
        surface_data_list = []
        with xr.open_dataset(single_file) as ds:
            for var in surface_vars:
                # 读取数据并确保转换为 float32
                data = ds[var][:].astype(np.float32) # 形状: (1, 721, 1440)
                downsampled = data.coarsen(latitude=4, longitude=4, boundary="trim").mean()
                surface_data_list.append(downsampled)
        # 拼接地面数据，结果为 float32
        surface_data = np.concatenate(surface_data_list, axis=0) # 形状: (4, 180, 360), 类型: float32

        # 合并upper和surface数据，结果为 float32
        data = np.concatenate([upper_data, surface_data], axis=0) # 形状: (69, 180, 360), 类型: float32

        # 进行归一化操作
        data = (data - avg) / std
        # 检查数据中是否存在无效值(NaN或Inf)
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        if nan_count > 0 or inf_count > 0:
            logger.info(f"警告：数据中存在无效值 - NaN: {nan_count}, Inf: {inf_count}")
            # 可选：将无效值替换为0或其他值
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            logger.info(f"已将无效值替换为0")
        
        # 转换为float16以节省存储空间
        data = data.astype(np.float16)
        if dt == datetime(1987, 1, 1, 0, 0):
            logger.info(f"mean: {np.mean(data)}")
            logger.info(f"std: {np.std(data)}")
        
        # 保存为npy格式
        # 根据时间dt创建目录结构并生成npy文件路径
        year_dir = os.path.join(ERA5_NORM_OUTPUT_ROOT, str(dt.year))
        date_dir = os.path.join(year_dir, dt.strftime('%Y%m%d'))
        # 确保目录存在
        os.makedirs(date_dir, exist_ok=True)
        # 生成最终的npy文件路径
        npy_path = os.path.join(date_dir, f"ERA5_Global_LM_{dt.strftime('%Y%m%d%H')}.npy")
        np.save(npy_path, data)
        
    except FileNotFoundError:
        logger.info(f"错误：数据文件未找到 for time {dt.strftime('%Y%m%d%H')}")
        # 根据需要返回 None 或引发异常
        # 返回 None 可能导致 __getitem__ 出错，需要进一步处理
        # raise # 或者直接重新引发异常，让 DataLoader 处理
        return None # 暂时返回 None，需要在 __getitem__ 中处理
    except KeyError:
        logger.info(f"错误：在 .npz 文件中未找到名为 'data' 的数组 for time {dt.strftime('%Y%m%d%H')}")
        # raise
        return None
    except Exception as e:
        logger.info(f"加载数据时发生错误 for time {dt.strftime('%Y%m%d%H')}: {e}")
        # raise
        return None
 
def main():
    import time
    start_year = 1979
    end_year = 2022
    avg, std = load_avg_std()
    logger = setup_logger(start_year, end_year)
    time_indices = _generate_time_indices(start_year, end_year)
    for dt in time_indices:
        logger.info(f"--------正在处理时间: {dt.strftime('%Y%m%d%H')}--------")
        start_time = time.time()
        _load_data_for_time(dt, avg, std, logger)
        end_time = time.time()
        logger.info(f"所用时间: {end_time - start_time}秒")
    logger.info("数据处理完成")

if __name__ == "__main__":
    main()
