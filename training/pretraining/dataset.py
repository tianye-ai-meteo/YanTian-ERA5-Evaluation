import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from config import data_dir
class BaselineDataset(Dataset):
    def __init__(self, root_dir=data_dir, start_year=1979, end_year=2016):
        """
        初始化数据集
        Args:
            start_year: 起始年份
            end_year: 结束年份
        """
        self.root_dir = root_dir
        self.start_year = start_year
        self.end_year = end_year

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
        hours = [0, 6, 12, 18]  # 每天需要的时刻
        
        for year in range(self.start_year, self.end_year + 1):
            # 对于起始年份的第一天特殊处理
            if year == self.start_year:
                start_date = datetime(year, 1, 1, 0)  # 从0点开始
                # 第一天只能用0点和6点预测12点，用6点和12点预测18点，用12点和18点预测次日0点
                first_day = start_date.date()
                current_date = start_date
                while current_date.date() == first_day:
                    if current_date.hour in [12, 18]:  # 第一天只收集12点和18点作为预测目标
                        past_time = current_date - timedelta(hours=12)
                        current_exists = os.path.exists(self._get_file_path(current_date - timedelta(hours=6)))
                        past_exists = os.path.exists(self._get_file_path(past_time))
                        target_exists = os.path.exists(self._get_file_path(current_date))
                        if past_exists and current_exists and target_exists:
                            time_indices.append(current_date)
                    current_date += timedelta(hours=6)
                start_date = datetime(year, 1, 2)  # 从第二天开始正常处理
            else:
                start_date = datetime(year, 1, 1)
            

            end_date = datetime(year, 12, 31, 18)

            # 处理正常情况
            current_date = start_date
            while current_date <= end_date:
                if current_date.hour in hours:
                    # 获取用于预测的两个时刻
                    if current_date.hour == 6:  # 预测6点需要前一天18点和今天0点
                        past_time = current_date - timedelta(hours=12)  # 前一天18点
                        if past_time.year < self.start_year:  # 跳过需要上一年数据的情况
                            current_date += timedelta(hours=6)
                            continue
                        current_exists = os.path.exists(self._get_file_path(current_date - timedelta(hours=6)))
                    elif current_date.hour == 0:  # 预测0点需要前一天12点和18点
                        past_time = current_date - timedelta(hours=12)  # 前一天12点
                        if past_time.year < self.start_year:  # 跳过需要上一年数据的情况
                            current_date += timedelta(hours=6)
                            continue
                        current_exists = os.path.exists(self._get_file_path(current_date - timedelta(hours=6)))
                    else:  # 预测12点和18点使用前6小时和当前时刻
                        past_time = current_date - timedelta(hours=12)
                        current_exists = os.path.exists(self._get_file_path(current_date - timedelta(hours=6)))
                    
                    # 检查所需的输入和目标数据是否存在
                    past_exists = os.path.exists(self._get_file_path(past_time))
                    target_exists = os.path.exists(self._get_file_path(current_date))
                    
                    if past_exists and current_exists and target_exists:
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
        try:
            # 加载pressure层数据
            data_file = self._get_file_path(dt)
            data = np.load(data_file) # (69, 180, 360)
            return torch.from_numpy(data)
        except FileNotFoundError:
            print(f"错误：数据文件未找到 for time {dt.strftime('%Y%m%d%H')}")
            return None # 暂时返回 None，需要在 __getitem__ 中处理
        except KeyError:
            print(f"错误：在 .npz 文件中未找到名为 'data' 的数组 for time {dt.strftime('%Y%m%d%H')}")
            return None
        except Exception as e:
            print(f"加载数据时发生错误 for time {dt.strftime('%Y%m%d%H')}: {e}")
            return None

    def __len__(self):
        """返回数据集长度"""
        return len(self.time_indices)

    def __getitem__(self, idx):
        """获取单个数据样本"""
        target_time = self.time_indices[idx]
        
        # 根据目标时间确定输入数据的时间点
        past_time = target_time - timedelta(hours=12)  # 前一天18时
        current_time = target_time - timedelta(hours=6)  # 00时
        
        # 加载数据，并处理可能的 None 返回值
        past_data = self._load_data_for_time(past_time)
        current_data = self._load_data_for_time(current_time)
        future_data = self._load_data_for_time(target_time)

        # 检查是否有数据加载失败
        if past_data is None or current_data is None or future_data is None:
            # 如果任何一个时间点的数据加载失败，则无法构成有效样本
            print(f"警告: 无法加载索引 {idx} (目标时间 {target_time}) 的完整数据，跳过...")
            return None # 或者根据你的 DataLoader 配置返回其他值

        # 将过去和当前数据在第0维度拼接，形成输入数据 (2, 69, 180, 360)
        # 让数据保持 float32，AMP 会处理精度转换
        input_data = torch.stack([past_data, current_data], dim=0) 
        
        # 输出数据 (69, 180, 360)
        output_data = future_data 
        
        
        # 返回输入和目标
        return input_data, output_data

class BaselineValDataset(BaselineDataset):
    """
    验证数据集类，继承自训练数据集的 BaselineDataset。
    默认使用2017年的数据。
    """
    def __init__(self, root_dir=data_dir, start_year=2017, end_year=2017):
        """
        初始化验证数据集。
        Args:
            root_dir (str): 数据根目录 (包含 nc 文件)。
            start_year (int): 起始年份，默认为2017。
            end_year (int): 结束年份，默认为2017。
        """
        # 调用父类的 __init__ 方法，并传递指定的年份
        super().__init__(root_dir=root_dir, start_year=start_year, end_year=end_year)


if __name__ == '__main__':
    dataset = BaselineDataset(start_year=1997, end_year=2016)
    # 检查第一个样本的形状
    if len(dataset) > 0:
        sample = dataset[0]
        if sample is not None:
            input_data, output_data = sample
            print(f"数据集长度: {len(dataset)}")
            print("第一个样本输入形状:", input_data.shape) # 预期: torch.Size([2, 69, 180, 360])
            print("第一个样本输入类型:", input_data.dtype) # 修正后预期: torch.float32
            print("第一个样本输出形状:", output_data.shape) # 预期: torch.Size([69, 180, 360])
            print("第一个样本输出类型:", output_data.dtype) # 修正后预期: torch.float32
            print("第一个样本切片:", input_data[0, 40, 10, 10]) # -1.0838
            print("第一个样本切片:", input_data.to(dtype=torch.float16)[0, 40, 10, 10]) # -1.0838
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
            print("批次输入形状:", batch_input.shape)  # 预期: torch.Size([3, 2, 69, 180, 360])
            print("批次输出形状:", batch_output.shape)  # 预期: torch.Size([3, 69, 180, 360])
            break
        else:
            print("无法加载批次数据。")
    '''
    预期输出:
    第一个样本输入形状: torch.Size([2, 69, 180, 360])
    第一个样本输入类型: torch.float16
    第一个样本输出形状: torch.Size([69, 180, 360])
    第一个样本输出类型: torch.float16
    '''
