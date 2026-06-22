import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn # 引入神经网络模块
import torch.optim as optim # 引入优化器模块

class LatitudeWeightedMAELoss(nn.Module):
    """
    带纬度加权和可选通道加权的平均绝对误差损失。
    权重与输入/目标张量自动匹配设备和数据类型。
    """
    def __init__(self, lat_dim=180, num_channels=69, enable_channel_weights=True):
        """
        Args:
            lat_dim (int): 纬度方向的格点数, 默认为 180。
            num_channels (int): 通道数，用于通道加权。默认为 69。
            enable_channel_weights (bool): 是否启用通道加权。默认为 False。
        """
        super().__init__()
        # 将纬度权重注册为 buffer
        latitude_weights = self.compute_latitude_weights(lat_dim)
        self.register_buffer('weights_lat', latitude_weights)

        self.enable_channel_weights = enable_channel_weights
        if self.enable_channel_weights:
            if num_channels != 69:
                print(f"警告: 通道加权目前仅为84个通道特别设计。对于 {num_channels} 个通道，将使用等权重。")
            # 将通道权重注册为 buffer
            channel_weights = self.compute_channel_weights(num_channels)
            self.register_buffer('weights_chan', channel_weights)

    def compute_latitude_weights(self, lat_dim):
        """计算纬度权重 (cos(latitude))"""
        # 生成纬度数组
        lats_degrees = np.linspace(-90, 90, lat_dim)
        # 将角度转换为弧度
        lats_rad = np.deg2rad(lats_degrees)
        # 计算余弦值作为权重
        cos_lats = np.cos(lats_rad)
        # 确保权重是非负的
        cos_lats = np.maximum(cos_lats, 0)
        # 转换为 PyTorch 张量并确保是 float32
        return torch.from_numpy(cos_lats).float()

    def compute_channel_weights(self, num_channels):
        """
        计算通道权重。
        权重是为总共84个通道硬编码的：
        - 78个大气变量 (z,r,t,u,v,w, 各13层) + 6个地面变量。
        - 权重设计:
          - Z, U, V 变量以及地面变量的权重更高。
          - 大气变量中，低层（高气压）的权重大于高层。
        """
        if num_channels != 69:
            return torch.ones(num_channels).float()

        # 1. 定义变量类型的基础乘数
        variable_multipliers = {'z': 2.0, 'r': 1.0, 't': 1.0, 'u': 1.0, 'v': 1.0}

        # --- 2. 为 'z' 变量创建自定义权重 ---
        # 压力层: [50,100,150,200,250,300,400,500,600,700,850,925,1000]
        # 对应索引: [0, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,  12]
        z_level_weights = torch.zeros(13)
        # 从50hPa(索引0)到500hPa(索引7)线性增加至最大值
        linear_part = torch.linspace(1.0, 1.0, 8) # 8个点，50hPa到500hPa
        z_level_weights[0:8] = linear_part
        # 根据规则设置其他高度层的权重
        weight_400hpa = z_level_weights[6] # 400hPa的权重
        weight_500hpa = z_level_weights[7] # 500hPa的权重 (最大值)
        # 600(索引8), 700(索引9), 925(索引11)的权重与400hpa相同
        z_level_weights[8] = weight_400hpa
        z_level_weights[9] = weight_400hpa
        z_level_weights[11] = weight_400hpa
        # 850(索引10), 1000(索引12)的权重与500hpa相同 (最大值)
        z_level_weights[10] = weight_500hpa
        z_level_weights[12] = weight_500hpa
        # 应用变量乘数
        z_weights = variable_multipliers['z'] * z_level_weights

        # --- 3. 为其他大气变量创建权重 ---
        pressure_level_weights = torch.linspace(1.0, 1.0, 13)
        
        weights = []
        weights.append(z_weights) # 添加 'z' 的权重
        # 变量顺序: z(已添加), r, t, u, v
        for var in ['r', 't', 'u', 'v']:
            multiplier = variable_multipliers[var]
            weights.append(multiplier * pressure_level_weights)
        
        # --- 4. 为地面变量创建权重 ---
        surface_weights = torch.full((4,), 1.0)
        weights.append(surface_weights)

        # --- 5. 组合所有权重并进行最终调整 ---
        channel_weights = torch.cat(weights)
        
        # 将u10、v10、msl、T2m 地面关键变量的权重设置为全局最高
        max_weight = torch.max(channel_weights)
        channel_weights[68] = max_weight  # msl (索引68)
        # channel_weights[67] = max_weight  # T2m (索引67)

        return channel_weights.float()

    def forward(self, input, target):
        """
        Args:
            input (torch.Tensor): 模型输出，形状 (B, C, H, W)。
            target (torch.Tensor): 目标值，形状与 input 相同。
        Returns:
            torch.Tensor: 加权 MAE 损失值 (标量)。
        """
        B, C, H, W = input.shape
        # 计算逐元素的绝对误差
        error = torch.abs(input - target)

        # --- 纬度权重 ---
        lat_weights = self.weights_lat.to(input.device, dtype=input.dtype).view(1, 1, -1, 1)

        # --- 合并权重并计算总权重 ---
        if self.enable_channel_weights:
            if C != len(self.weights_chan):
                 raise ValueError(f"输入通道维度 ({C}) 与权重通道维度 ({len(self.weights_chan)}) 不匹配。")
            chan_weights = self.weights_chan.to(input.device, dtype=input.dtype).view(1, -1, 1, 1)
            
            # 合并纬度和通道权重
            weights = lat_weights * chan_weights # Broadcasting to (1, C, H, 1)
            sum_of_weights = torch.sum(self.weights_lat) * torch.sum(self.weights_chan) * B * W
        else:
            weights = lat_weights
            sum_of_weights = torch.sum(self.weights_lat) * B * C * W

        # --- 应用权重并计算损失 ---
        weighted_error = error * weights
        sum_of_weighted_error = torch.sum(weighted_error)
        
        epsilon = 1e-12
        if sum_of_weights > epsilon:
            weighted_mae = sum_of_weighted_error / sum_of_weights
        else:
            weighted_mae = torch.mean(error) # Fallback to unweighted

        return weighted_mae

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

# class DynamicWeightedMAELoss(nn.Module):
#     """
#     带纬度加权和动态通道加权的平均绝对误差损失。

#     该损失函数实现了两个核心功能：
#     1. 动态权重调整：根据每个变量的指数移动平均（EMA）损失，
#        自动计算通道权重，使得损失较大的变量获得较小的权重，反之亦然，
#        从而平衡不同变量对总损失的贡献。
#     2. 详细损失与权重监控：在训练过程中，打印每个物理变量的当前批次平均损失，
#        以及根据该损失更新后的、将用于下一次迭代的新权重。
#     """
#     def __init__(self, 
#                  lat_dim: int = 180,
#                  ema_decay: float = 0.9, 
#                  epsilon: float = 1e-6,
#                  print_losses: bool = True):
#         """
#         Args:
#             lat_dim (int): 纬度方向的格点数, 默认为 721。
#             ema_decay (float): 指数移动平均的衰减率。用于平滑损失值。值越接近1，平滑效果越强。
#             epsilon (float): 一个小常数，用于防止计算权重时除以零。
#             print_losses (bool): 是否在每次调用forward时打印各变量损失与权重。
#         """
#         super().__init__()
#         self.ema_decay = ema_decay
#         self.epsilon = epsilon
#         self.print_losses = print_losses
        
#         # 1. 定义变量及其对应的通道索引
#         self.variable_groups = OrderedDict([
#             # 高空多层变量 (名称，起始索引，层数)
#             ('z_50-1000hPa', (0, 13)),
#             ('r_50-1000hPa', (13, 13)),
#             ('t_50-1000hPa', (26, 13)),
#             ('u_50-1000hPa', (39, 13)),
#             ('v_50-1000hPa', (52, 13)),
#             # 低空单层变量 (名称，起始索引，层数)
#             ('u10m', (65, 1)),
#             ('v10m', (66, 1)),
#             ('t2m', (67, 1)),
#             ('msl', (68, 1)),
#         ])
        
#         self.num_variables = len(self.variable_groups)
#         self.num_channels = sum(v[1] for v in self.variable_groups.values()) # 应为84

#         # 2. 注册纬度权重
#         latitude_weights = self.compute_latitude_weights(lat_dim)
#         self.register_buffer('weights_lat', latitude_weights)

#         # 3. 注册用于存储EMA损失的buffer
#         running_mean_loss = torch.ones(self.num_variables)
#         self.register_buffer('running_mean_loss', running_mean_loss)
        
#         # 4. 注册通道权重buffer，将在forward中动态更新
#         channel_weights = torch.ones(self.num_channels)
#         self.register_buffer('weights_chan', channel_weights)


#     def compute_latitude_weights(self, lat_dim: int) -> torch.Tensor:
#         """计算纬度权重 (cos(latitude))"""
#         lats_degrees = np.linspace(-90, 90, lat_dim)
#         lats_rad = np.deg2rad(lats_degrees)
#         cos_lats = np.cos(lats_rad)
#         cos_lats = np.maximum(cos_lats, 0)
#         return torch.from_numpy(cos_lats).float()

#     def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         计算加权损失。
#         Args:
#             prediction (torch.Tensor): 模型预测值，维度 (B, C, H, W)。
#             target (torch.Tensor): 真实值，维度 (B, C, H, W)。
        
#         Returns:
#             torch.Tensor: 一个标量损失值。
#         """
#         assert prediction.shape == target.shape
#         assert prediction.dim() == 4
#         assert prediction.shape[1] == self.num_channels
        
#         # --- 步骤 1: 计算原始MAE并应用纬度权重 ---
#         error = torch.abs(prediction - target)
#         lat_weights_reshaped = self.weights_lat.to(error.device, error.dtype).view(1, 1, -1, 1)
#         lat_weighted_error = error * lat_weights_reshaped
        
#         # --- 步骤 2: 计算每个变量的当前批次平均损失 ---
#         per_channel_loss = torch.mean(lat_weighted_error, dim=(0, 2, 3))
#         per_variable_loss = torch.zeros(self.num_variables, device=error.device, dtype=error.dtype)
#         for i, (name, (start_idx, num_levels)) in enumerate(self.variable_groups.items()):
#             variable_loss = torch.mean(per_channel_loss[start_idx : start_idx + num_levels])
#             per_variable_loss[i] = variable_loss.detach()

#         # --- 步骤 3: 更新EMA损失并计算新的权重 (仅在训练模式下) ---
#         normalized_variable_weights = None
#         if self.training:
#             # 更新EMA损失
#             self.running_mean_loss = (self.ema_decay * self.running_mean_loss +
#                                       (1 - self.ema_decay) * per_variable_loss)

#             # 根据EMA损失计算每个变量的权重 (与损失成反比)
#             variable_weights = 1.0 / (self.running_mean_loss + self.epsilon)
            
#             # 归一化权重，使其总和等于变量数，以保持总损失的量级稳定 (即平均权重为1)
#             normalized_variable_weights = variable_weights * (self.num_variables / torch.sum(variable_weights))
            
#             # 将变量权重扩展到每个通道上并更新buffer
#             new_channel_weights = torch.zeros_like(self.weights_chan)
#             for i, (name, (start_idx, num_levels)) in enumerate(self.variable_groups.items()):
#                 new_channel_weights[start_idx : start_idx + num_levels] = normalized_variable_weights[i]
#             self.weights_chan.copy_(new_channel_weights)

#         # --- 新增: 步骤 3.5: 打印损失和对应的权重 ---
#         if self.print_losses:
#             print_buffer = "--- 变量损失与对应权重 ---\n"
#             header = f"{'变量':<15} | {'当前批次损失':<18} | {'新权重 (用于下次迭代)':<25}"
#             print_buffer += header + "\n"
#             print_buffer += "-" * (len(header) + 2) + "\n"

#             for i, name in enumerate(self.variable_groups.keys()):
#                 current_loss_val = per_variable_loss[i].item()
                
#                 if self.training:
#                     # 在训练模式下，打印刚刚计算出的新权重
#                     new_weight_val = normalized_variable_weights[i].item()
#                     weight_str = f"{new_weight_val:.4f}"
#                 else:
#                     # 在评估模式下，权重是固定的，打印当前使用的权重
#                     start_idx, _ = self.variable_groups[name]
#                     static_weight_val = self.weights_chan[start_idx].item()
#                     weight_str = f"{static_weight_val:.4f} (评估模式，权重固定)"

#                 print_buffer += f"{name:<15} | {current_loss_val:<18.6f} | {weight_str}\n"
#             print(print_buffer.strip())


#         # --- 步骤 4: 应用通道权重并计算最终损失 ---
#         # 注意：这里使用的是 self.weights_chan，它在训练模式下刚刚被更新
#         chan_weights_reshaped = self.weights_chan.to(error.device, error.dtype).view(1, -1, 1, 1)
#         fully_weighted_error = lat_weighted_error * chan_weights_reshaped
#         final_loss = torch.mean(fully_weighted_error)
        
#         return final_loss
 
class DynamicWeightedMAELoss(nn.Module):
    def __init__(self, 
                 lat_dim: int = 180,
                 ema_decay: float = 0.9, 
                 epsilon: float = 1e-6,
                 print_losses: bool = True):
        super().__init__()
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.print_losses = print_losses

        self.variable_groups = OrderedDict([
            ('z_50-1000hPa', (0, 13)),
            ('r_50-1000hPa', (13, 13)),
            ('t_50-1000hPa', (26, 13)),
            ('u_50-1000hPa', (39, 13)),
            ('v_50-1000hPa', (52, 13)),
            ('u10m', (65, 1)),
            ('v10m', (66, 1)),
            ('t2m', (67, 1)),
            ('msl', (68, 1)),
        ])

        self.num_variables = len(self.variable_groups)
        self.num_channels = sum(v[1] for v in self.variable_groups.values())

        latitude_weights = self.compute_latitude_weights(lat_dim)
        self.register_buffer('weights_lat', latitude_weights)

        running_mean_loss = torch.ones(self.num_variables)
        self.register_buffer('running_mean_loss', running_mean_loss)

        channel_weights = torch.ones(self.num_channels)
        self.register_buffer('weights_chan', channel_weights)

    def compute_latitude_weights(self, lat_dim: int) -> torch.Tensor:
        lats_degrees = np.linspace(-90, 90, lat_dim)
        lats_rad = np.deg2rad(lats_degrees)
        cos_lats = np.cos(lats_rad)
        cos_lats = np.maximum(cos_lats, 0)
        return torch.from_numpy(cos_lats).float()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert prediction.shape == target.shape
        assert prediction.dim() == 4
        assert prediction.shape[1] == self.num_channels

        error = torch.abs(prediction - target)
        lat_weights_reshaped = self.weights_lat.to(error.device, error.dtype).view(1, 1, -1, 1)
        lat_weighted_error = error * lat_weights_reshaped

        per_channel_loss = torch.mean(lat_weighted_error, dim=(0, 2, 3))
        per_variable_loss = torch.zeros(self.num_variables, device=error.device, dtype=error.dtype)
        for i, (name, (start_idx, num_levels)) in enumerate(self.variable_groups.items()):
            variable_loss = torch.mean(per_channel_loss[start_idx : start_idx + num_levels])
            per_variable_loss[i] = variable_loss.detach()  # 切断梯度

        normalized_variable_weights = None
        if self.training:
            # --- 注意：更新 buffer 时要在 no_grad 下 ---
            with torch.no_grad():
                self.running_mean_loss.mul_(self.ema_decay).add_(
                    (1 - self.ema_decay) * per_variable_loss
                )

                variable_weights = 1.0 / (self.running_mean_loss + self.epsilon)
                normalized_variable_weights = variable_weights * (
                    self.num_variables / torch.sum(variable_weights)
                )

                new_channel_weights = torch.zeros_like(self.weights_chan)
                for i, (name, (start_idx, num_levels)) in enumerate(self.variable_groups.items()):
                    new_channel_weights[start_idx : start_idx + num_levels] = normalized_variable_weights[i]
                self.weights_chan.copy_(new_channel_weights)

        if self.print_losses:
            print_buffer = "--- 变量损失与对应权重 ---\n"
            header = f"{'变量':<15} | {'当前批次损失':<18} | {'新权重 (用于下次迭代)':<25}"
            print_buffer += header + "\n"
            print_buffer += "-" * (len(header) + 2) + "\n"
            for i, name in enumerate(self.variable_groups.keys()):
                current_loss_val = per_variable_loss[i].item()
                if self.training:
                    new_weight_val = normalized_variable_weights[i].item()
                    weight_str = f"{new_weight_val:.4f}"
                else:
                    start_idx, _ = self.variable_groups[name]
                    static_weight_val = self.weights_chan[start_idx].item()
                    weight_str = f"{static_weight_val:.4f} (eval固定)"
                print_buffer += f"{name:<15} | {current_loss_val:<18.6f} | {weight_str}\n"
            print(print_buffer.strip())

        # --- 关键修复：clone 出一份干净的权重参与乘法 ---
        chan_weights_reshaped = self.weights_chan.to(error.device, error.dtype).view(1, -1, 1, 1).clone()
        fully_weighted_error = lat_weighted_error * chan_weights_reshaped
        final_loss = torch.mean(fully_weighted_error)
        return final_loss
