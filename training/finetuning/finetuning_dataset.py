"""
Author: lity
Date: 2025-07-28
Description: 
    文件功能描述: 演天模型微调数据集
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
from datetime import datetime, timedelta
from config import *
import time

class BaselineDataset(Dataset):
    def __init__(self, root_dir=data_dir, start_year=1987, end_year=2016, step=6):
        """
        初始化数据集
        Args:
            root_dir: 数据根目录 (包含 npz 文件)
            start_year: 起始年份
            end_year: 结束年份
        """
        self.root_dir = root_dir
        self.start_year = start_year
        self.end_year = end_year
        self.step = step

        
        # 生成时间索引列表
        self.time_indices = self._generate_time_indices()
        
    def _generate_time_indices(self):
        """生成时间索引列表
        每天的采样时刻为00、06、12、18
        预测模式为：
        1. 18,00 -> 06 (对起始年份第一天不可用)
        2. 00,06 -> 12
        3. 06,12 -> 18
        4. 12,18 -> 次日00
        """

        time_indices = []
        
        for year in range(self.start_year, self.end_year + 1):
            start_date = datetime(year, 1, 1, 0)
            end_date = datetime(year, 12, 31, 18)
            current_date = start_date
            
            while current_date <= end_date:
                past_time = current_date - timedelta(hours=6)
                
                # 检查所有需要的文件是否存在
                all_files_exist = True
                # 检查过去和当前的文件
                if not os.path.exists(self._get_file_path(past_time)) or \
                   not os.path.exists(self._get_file_path(current_date)):
                    all_files_exist = False

                # 检查所有未来的文件直到 self.step
                if all_files_exist:
                    for i in range(self.step):
                        future_time = current_date + timedelta(hours=6 * (i + 1))
                        if not os.path.exists(self._get_file_path(future_time)):
                            all_files_exist = False
                            break  # 如果有一个缺失，就没必要继续检查了
                
                if all_files_exist:
                    time_indices.append(current_date)
                
                current_date += timedelta(hours=6)

        return sorted(time_indices)  # 确保时间顺序
        


    def _get_file_path(self, dt):
        """获取指定时间和类型的文件路径 (npy格式)
        Args:
            dt (datetime): 时间点
        Returns:
            str: 文件完整路径
        """
        year_dir = os.path.join(self.root_dir, str(dt.year))
        date_dir = os.path.join(year_dir, dt.strftime('%Y%m%d'))

        filename = f'ERA5_Global_LM_{dt.strftime("%Y%m%d%H")}.npy' # <--- 修改后缀

        return os.path.join(date_dir, filename)


    def _load_data_for_time(self, dt):
        """加载指定时间的数据 (从 .nc 文件)，返回(84, 721, 1440) 的 torch.float32 张量
            1、加载pressure层数据和single层数据，并合并为(84, 721, 1440)
                0-12: z[1-13],50hpa-1000hpa
                13-25: r[1-13],50hpa-1000hpa
                26-38: t[1-13],50hpa-1000hpa
                39-51: u[1-13],50hpa-1000hpa
                52-64: v[1-13],50hpa-1000hpa
                65-77: w[1-13],50hpa-1000hpa
                78-83: u10,v10,d2m,t2m,msl,sp
            2、使用归一化数据进行整体归一化
            3、(合并常量数据已移至 __getitem__ 中)
            4、返回 torch.float32 张量
        """

        try:
            # 加载pressure层数据
            data_file = self._get_file_path(dt)
            data = np.load(data_file) # (84, 721, 1440)


            return torch.from_numpy(data)
        except FileNotFoundError:
            print(f"错误：数据文件未找到 for time {dt.strftime('%Y%m%d%H')}")
            # 根据需要返回 None 或引发异常
            # 返回 None 可能导致 __getitem__ 出错，需要进一步处理
            # raise # 或者直接重新引发异常，让 DataLoader 处理
            return None # 暂时返回 None，需要在 __getitem__ 中处理
        except KeyError:
            print(f"错误：在 .npz 文件中未找到名为 'data' 的数组 for time {dt.strftime('%Y%m%d%H')}")
            # raise
            return None
        except Exception as e:
            print(f"加载数据时发生错误 for time {dt.strftime('%Y%m%d%H')}: {e}")
            # raise
            return None

    def __len__(self):
        """返回数据集长度"""
        return len(self.time_indices)

    def __getitem__(self, idx):
        """获取单个数据样本"""
        current_time = self.time_indices[idx]

        # 生成输入
        past_time = current_time - timedelta(hours=6)  # 前一天18时
        past_data = self._load_data_for_time(past_time)

        current_data = self._load_data_for_time(current_time)

        input_data = torch.stack([past_data, current_data], dim=0) 

        # 根据迭代步数，生成多个标签和对应的未来时间特征
        labels = []
        for i in range(self.step):
            future_time = current_time + timedelta(hours=6*(i+1))
            future_data = self._load_data_for_time(future_time)
            labels.append(future_data)


        return input_data, labels

if __name__ == '__main__':
    dataset = BaselineDataset(start_year=2017, end_year=2018, step=6)
    # 检查第一个样本的形状
    if len(dataset) > 0:
        sample = dataset[0]
        if sample is not None:
            input_data, labels = sample
            print(f"数据集长度: {len(dataset)}")
            print("第一个样本输入形状:", input_data.shape) # 预期: torch.Size([2, 92, 721, 1440])
            print("第一个样本输入类型:", input_data.dtype) # 修正后预期: torch.float32
            print("第一个样本输出形状:", labels[0].shape) # 预期: torch.Size([84, 721, 1440])
            print("第一个样本输出类型:", labels[0].dtype) # 修正后预期: torch.float32
            print("第一个样本切片:", input_data[0, 40, 40, 40]) # -1.0838
            print("第一个样本切片:", input_data.to(dtype=torch.float16)[0, 40, 30, 20]) # -1.0838
        else:
            print("无法加载第一个样本。")
    else:
        print("数据集为空。")
    
    # 使用DataLoader加载数据，batch_size=3
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True)
    
    # 获取第一个批次并打印维度
    for batch in dataloader:
        if batch is not None:
            batch_input, batch_output = batch
            print("\n使用DataLoader加载的第一个批次:")
            print("批次输入形状:", batch_input.shape)  # 预期: torch.Size([3, 2, 92, 721, 1440])
            print("批次输出形状:", batch_output[0].shape)  # 预期: torch.Size([3, 84, 721, 1440])
            print("标签长度:", len(batch_output))
            break
        else:
            print("无法加载批次数据。")
    '''
    预期输出:
    第一个样本输入形状: torch.Size([2, 96, 721, 1440])
    第一个样本输入类型: torch.float16
    第一个样本输出形状: torch.Size([84, 721, 1440])
    第一个样本输出类型: torch.float16
    '''