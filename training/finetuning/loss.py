import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn 
import torch.optim as optim 

class LatitudeWeightedMAELoss(nn.Module):
    def __init__(self, lat_dim=180, num_channels=69, enable_channel_weights=True):
        """Create a latitude-weighted MAE loss with optional channel weights."""
        super().__init__()
        
        latitude_weights = self.compute_latitude_weights(lat_dim)
        self.register_buffer('weights_lat', latitude_weights)

        self.enable_channel_weights = enable_channel_weights
        if self.enable_channel_weights:
            if num_channels != 69:
                print(f"Warning: channel weighting is currently designed for 84 channels. Equal weights will be used for {num_channels} channels.")
            
            channel_weights = self.compute_channel_weights(num_channels)
            self.register_buffer('weights_chan', channel_weights)

    def compute_latitude_weights(self, lat_dim):
        """Compute cosine latitude weights for a regular latitude grid."""
        
        lats_degrees = np.linspace(-90, 90, lat_dim)
        
        lats_rad = np.deg2rad(lats_degrees)
        
        cos_lats = np.cos(lats_rad)
        
        cos_lats = np.maximum(cos_lats, 0)
        
        return torch.from_numpy(cos_lats).float()

    def compute_channel_weights(self, num_channels):
        """Build channel weights for the 69-channel ERA5 variable layout."""
        if num_channels != 69:
            return torch.ones(num_channels).float()

        
        variable_multipliers = {'z': 2.0, 'r': 1.0, 't': 1.0, 'u': 1.0, 'v': 1.0}

        
        
        
        z_level_weights = torch.zeros(13)
        
        linear_part = torch.linspace(1.0, 1.0, 8) 
        z_level_weights[0:8] = linear_part
        
        weight_400hpa = z_level_weights[6] 
        weight_500hpa = z_level_weights[7] 
        
        z_level_weights[8] = weight_400hpa
        z_level_weights[9] = weight_400hpa
        z_level_weights[11] = weight_400hpa
        
        z_level_weights[10] = weight_500hpa
        z_level_weights[12] = weight_500hpa
        
        z_weights = variable_multipliers['z'] * z_level_weights

        
        pressure_level_weights = torch.linspace(1.0, 1.0, 13)
        
        weights = []
        weights.append(z_weights) 
        
        for var in ['r', 't', 'u', 'v']:
            multiplier = variable_multipliers[var]
            weights.append(multiplier * pressure_level_weights)
        
        
        surface_weights = torch.full((4,), 1.0)
        weights.append(surface_weights)

        
        channel_weights = torch.cat(weights)
        
        
        max_weight = torch.max(channel_weights)
        channel_weights[68] = max_weight  
        

        return channel_weights.float()

    def forward(self, input, target):
        """Return weighted mean absolute error between input and target."""
        B, C, H, W = input.shape
        
        error = torch.abs(input - target)

        
        lat_weights = self.weights_lat.to(input.device, dtype=input.dtype).view(1, 1, -1, 1)

        
        if self.enable_channel_weights:
            if C != len(self.weights_chan):
                 raise ValueError(f"Input channel dimension ({C}) does not match weight channel dimension ({len(self.weights_chan)}).")
            chan_weights = self.weights_chan.to(input.device, dtype=input.dtype).view(1, -1, 1, 1)
            
            
            weights = lat_weights * chan_weights # Broadcasting to (1, C, H, 1)
            sum_of_weights = torch.sum(self.weights_lat) * torch.sum(self.weights_chan) * B * W
        else:
            weights = lat_weights
            sum_of_weights = torch.sum(self.weights_lat) * B * C * W

        
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








#     """
#     def __init__(self, 
#                  lat_dim: int = 180,
#                  ema_decay: float = 0.9, 
#                  epsilon: float = 1e-6,
#                  print_losses: bool = True):
#         """
#         Args:




#         """
#         super().__init__()
#         self.ema_decay = ema_decay
#         self.epsilon = epsilon
#         self.print_losses = print_losses
        

#         self.variable_groups = OrderedDict([

#             ('z_50-1000hPa', (0, 13)),
#             ('r_50-1000hPa', (13, 13)),
#             ('t_50-1000hPa', (26, 13)),
#             ('u_50-1000hPa', (39, 13)),
#             ('v_50-1000hPa', (52, 13)),

#             ('u10m', (65, 1)),
#             ('v10m', (66, 1)),
#             ('t2m', (67, 1)),
#             ('msl', (68, 1)),
#         ])
        
#         self.num_variables = len(self.variable_groups)



#         latitude_weights = self.compute_latitude_weights(lat_dim)
#         self.register_buffer('weights_lat', latitude_weights)


#         running_mean_loss = torch.ones(self.num_variables)
#         self.register_buffer('running_mean_loss', running_mean_loss)
        

#         channel_weights = torch.ones(self.num_channels)
#         self.register_buffer('weights_chan', channel_weights)


#     def compute_latitude_weights(self, lat_dim: int) -> torch.Tensor:

#         lats_degrees = np.linspace(-90, 90, lat_dim)
#         lats_rad = np.deg2rad(lats_degrees)
#         cos_lats = np.cos(lats_rad)
#         cos_lats = np.maximum(cos_lats, 0)
#         return torch.from_numpy(cos_lats).float()

#     def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """

#         Args:


        
#         Returns:

#         """
#         assert prediction.shape == target.shape
#         assert prediction.dim() == 4
#         assert prediction.shape[1] == self.num_channels
        

#         error = torch.abs(prediction - target)
#         lat_weights_reshaped = self.weights_lat.to(error.device, error.dtype).view(1, 1, -1, 1)
#         lat_weighted_error = error * lat_weights_reshaped
        

#         per_channel_loss = torch.mean(lat_weighted_error, dim=(0, 2, 3))
#         per_variable_loss = torch.zeros(self.num_variables, device=error.device, dtype=error.dtype)
#         for i, (name, (start_idx, num_levels)) in enumerate(self.variable_groups.items()):
#             variable_loss = torch.mean(per_channel_loss[start_idx : start_idx + num_levels])
#             per_variable_loss[i] = variable_loss.detach()


#         normalized_variable_weights = None
#         if self.training:

#             self.running_mean_loss = (self.ema_decay * self.running_mean_loss +
#                                       (1 - self.ema_decay) * per_variable_loss)


#             variable_weights = 1.0 / (self.running_mean_loss + self.epsilon)
            

#             normalized_variable_weights = variable_weights * (self.num_variables / torch.sum(variable_weights))
            

#             new_channel_weights = torch.zeros_like(self.weights_chan)
#             for i, (name, (start_idx, num_levels)) in enumerate(self.variable_groups.items()):
#                 new_channel_weights[start_idx : start_idx + num_levels] = normalized_variable_weights[i]
#             self.weights_chan.copy_(new_channel_weights)


#         if self.print_losses:


#             print_buffer += header + "\n"
#             print_buffer += "-" * (len(header) + 2) + "\n"

#             for i, name in enumerate(self.variable_groups.keys()):
#                 current_loss_val = per_variable_loss[i].item()
                
#                 if self.training:

#                     new_weight_val = normalized_variable_weights[i].item()
#                     weight_str = f"{new_weight_val:.4f}"
#                 else:

#                     start_idx, _ = self.variable_groups[name]
#                     static_weight_val = self.weights_chan[start_idx].item()


#                 print_buffer += f"{name:<15} | {current_loss_val:<18.6f} | {weight_str}\n"
#             print(print_buffer.strip())




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
            per_variable_loss[i] = variable_loss.detach()  

        normalized_variable_weights = None
        if self.training:
            
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
            print_buffer = "--- Variable losses and corresponding weights ---\n"
            header = f"{'Variable':<15} | {'Current batch loss':<18} | {'New weight for next iteration':<25}"
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
                    weight_str = f"{static_weight_val:.4f} (fixed in eval mode)"
                print_buffer += f"{name:<15} | {current_loss_val:<18.6f} | {weight_str}\n"
            print(print_buffer.strip())

        
        chan_weights_reshaped = self.weights_chan.to(error.device, error.dtype).view(1, -1, 1, 1).clone()
        fully_weighted_error = lat_weighted_error * chan_weights_reshaped
        final_loss = torch.mean(fully_weighted_error)
        return final_loss
