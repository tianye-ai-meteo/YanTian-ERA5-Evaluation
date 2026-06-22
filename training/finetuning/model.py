import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple
from einops import rearrange
import numpy as np
from torch.utils.checkpoint import checkpoint
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 mlp_fc2_bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=mlp_fc2_bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_fc2_bias=True, init_std=0.02, pretrained_window_size=0,
                ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert (0 <= self.shift_size[0] < self.window_size[0]) and (0 <= self.shift_size[1] < self.window_size[1]), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           mlp_fc2_bias=mlp_fc2_bias)

        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        shortcut = x
        x = shortcut + self.drop_path(self.norm2(self.mlp(x)))
        return x
       
class PatchMerging(nn.Module):
    """
    Patch Merging Layer.
    将输入特征图的空间维度减半，通道维度调整为 out_dim。

    参数:
        input_resolution (tuple[int]): 输入特征图的分辨率 (H, W).
        in_dim (int): 输入特征图的通道数.
        out_dim (int): 输出特征图的通道数.
        norm_layer (nn.Module, optional): 归一化层. 默认: nn.LayerNorm
    """
    # 修改 __init__ 参数，接收 in_dim 和 out_dim
    def __init__(self, input_resolution, in_dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution # (H, W)
        self.in_dim = in_dim   # 输入通道 C
        self.out_dim = out_dim # 输出通道 C_out
        # 线性层，将 4*C 降维到 C_out
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)
        # 归一化层作用于合并后的 4*C 特征
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        """
        前向传播函数.
        输入 x 的形状: (B, H*W, C_in)
        输出 x 的形状: (B, H/2*W/2, C_out)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征图的长度与分辨率不匹配"
        assert C == self.in_dim, f"输入通道数 {C} 与期望的 in_dim {self.in_dim} 不匹配"
        assert H % 2 == 0 and W % 2 == 0, f"输入分辨率 ({H}*{W}) 不是偶数."

        # 将输入 reshape 为 (B, H, W, C_in)
        x = x.view(B, H, W, C)

        # 从输入的四个角落采样，实现空间降采样
        # x0: 左上角 (B, H/2, W/2, C_in)
        x0 = x[:, 0::2, 0::2, :]
        # x1: 左下角 (B, H/2, W/2, C_in)
        x1 = x[:, 1::2, 0::2, :]
        # x2: 右上角 (B, H/2, W/2, C_in)
        x2 = x[:, 0::2, 1::2, :]
        # x3: 右下角 (B, H/2, W/2, C_in)
        x3 = x[:, 1::2, 1::2, :]
        # 将四个角落的特征在通道维度拼接 (B, H/2, W/2, 4*C_in)
        x = torch.cat([x0, x1, x2, x3], -1)

        # 将拼接后的特征展平 (B, H/2*W/2, 4*C_in)
        x = x.view(B, -1, 4 * C)

        # 应用线性降维层 (B, H/2*W/2, C_out)
        x = self.reduction(x)
        # 归一化
        x = self.norm(x)

        return x
    
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, dim, bias=False)
        self.norm = norm_layer(dim//4)

    def forward(self, x):
        """
        x: B, H*W, C
        output: B, H*W*4, C//4
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels=69, out_channels=384):
        super().__init__()
        self.conv3d_block = nn.Conv3d(
                in_channels=in_channels,      # 输入通道数 69
                out_channels=out_channels, # 中间通道数 512
                kernel_size=(2, 2, 2), # 卷积核 (D, H, W)
                stride=(1, 2, 2),             # 步长 (D, H, W)
                padding=(0, 0, 0))           # 填充 (D, H, W)

        # 实例归一化 (作用于通道维度 C=512)
        self.layer_norm = nn.LayerNorm(out_channels) 
        self.gelu = nn.GELU()


    def forward(self, x):
        # 0. Permute -> (B, 69, 2, 180, 360)
        x = x.permute(0, 2, 1, 3, 4)
        # 1. Conv3d Block -> (B, 384, 90, 180)
        x = self.conv3d_block(x)
        # 2. Flatten -> (B, 384, 90*180)
        x = x.view(x.size(0), x.size(1), -1)
        # 3. transpose(B, 90*180, 384)
        x = x.transpose(1, 2)
        # 4. LayerNorm -> (B, 90*180, 384)
        x = self.layer_norm(x)
        # 5. GELU -> (B, 90*180, 384)
        x = self.gelu(x)
        shortcut_downsample = x # (B, 90*180, 384)
        return x, shortcut_downsample

class Encoder_High(nn.Module):
    def __init__(self, 
                 # transformer参数
                 depth=2,
                 in_channels=384, 
                 input_resolution=(90, 180),
                 num_heads=12, window_size=(6, 12), shift_size=(3, 6), # head_dim=32
                 mlp_ratio=4.,
                 drop=0.1, attn_drop=0.1, drop_path=0.1,
                 norm_layer=nn.LayerNorm,
                 # patch_merging参数
                 out_channels=768, 
                 # 使用checkpoints
                 use_checkpoints=True,
                 ):
        super().__init__()
        # 使用transformer进行处理
        self.use_checkpoints = use_checkpoints
        self.blocks = nn.ModuleList([SwinTransformerBlock(
            dim=in_channels, 
            input_resolution=input_resolution, 
            num_heads=num_heads, 
            window_size=window_size, shift_size=(0, 0) if (i % 2 == 0) else shift_size, 
            mlp_ratio=mlp_ratio, 
            drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer, 
            
            ) for i in range(depth)])
        # 空间维度减半，通道数加倍。数据维度从(B, 90, 180, C) 处理到 (B, 45, 90, 2C)
        self.patch_merging = PatchMerging(input_resolution=input_resolution, in_dim=in_channels, out_dim=out_channels)
            
    def forward(self, x):
        # swin_transformer_block
        for block in self.blocks:
            if self.use_checkpoints and block.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # 下采样
        x = self.patch_merging(x)
        shortcut_encoder_high = x # (B, 45*90, 768)
        return x, shortcut_encoder_high

class Core(nn.Module):
    def __init__(self, 
                 # transformer参数
                 # 常规参数
                 depth=30,
                 in_channels=768, 
                 input_resolution=(45, 90),
                 num_heads=24, window_size=(5, 10), shift_size=(2, 5), # head_dim=64
                 mlp_ratio=4.,
                 drop=0.1, attn_drop=0.1, drop_path=0.1,
                 norm_layer=nn.LayerNorm,
                 # 使用checkpoints
                 use_checkpoints=True,
                 ):
        super().__init__()
        self.dim = in_channels
        self.input_resolution = input_resolution
        self.use_checkpoints = use_checkpoints
        self.depth = depth

        # 定义指定深度的 Swin Transformer Block
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_params = {
                'dim': in_channels,
                'input_resolution': input_resolution,
                'num_heads': num_heads,
                'window_size': window_size,
                'shift_size': (0, 0) if (i % 2 == 0) else shift_size, # 保持交替移位
                'mlp_ratio': mlp_ratio,
                'drop': drop, 'attn_drop': attn_drop, 'drop_path': drop_path,
                'norm_layer': norm_layer,
            }
        
            self.blocks.append(SwinTransformerBlock(**block_params))

    def forward(self, x):
        # 确保输入是 (B, L, C)
        B, L, C = x.shape
        H, W = self.input_resolution
        assert C == self.dim and L == H * W, f"Core 输入形状错误: 期望 (B, {self.dim}, {H*W}), 得到 {x.shape}"

        # 应用 Swin Transformer Blocks
        for i, blk in enumerate(self.blocks):
            # SwinTransformerBlock 期望 (B, L, C) 输入
            if self.use_checkpoints and blk.training:
                output = checkpoint(blk, x, use_reentrant=False)
            else:
                output = blk(x)
            
        return output

class Decoder_High(nn.Module):
    def __init__(self, 
                 # 上采样Upsample参数
                 in_channels=768,
                 out_channels=384,
                 core_resolution=(45, 90),

                 # transformer参数
                 depth=2, 
                 input_resolution=(90, 180),
                 num_heads=12, window_size=(6, 12), shift_size=(3, 6), # head_dim=32
                 mlp_ratio=4.,
                 drop=0.1, attn_drop=0.1, drop_path=0.1,
                 norm_layer=nn.LayerNorm,
                 # 使用checkpoints
                 use_checkpoints=True,
                 ):
        super().__init__()
        self.core_resolution = core_resolution
        self.out_channels = out_channels
        self.use_checkpoints = use_checkpoints
        # 上采样
        self.patch_expand = PatchExpand(input_resolution=core_resolution, dim=in_channels*2, norm_layer=norm_layer)

        # transformer
        self.blocks = nn.ModuleList([SwinTransformerBlock(
            dim=out_channels, 
            input_resolution=input_resolution, 
            num_heads=num_heads, 
            window_size=window_size, shift_size=(0, 0) if (i % 2 == 0) else shift_size, 
            mlp_ratio=mlp_ratio, 
            drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer, 
            ) for i in range(depth)])
        
    def forward(self, x, shortcut_encoder_high):
        # 拼接shortcut_encoder_high与x
        x = torch.cat([x, shortcut_encoder_high], dim=-1) # (B, 45*90, 768*2)

        # patch_expand上采样
        x = self.patch_expand(x) # (B, 90*180, 384)
        
        # transformer
        for block in self.blocks:
            if self.use_checkpoints and block.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

class UpsampleBlock(nn.Module):
    '''
    将0.5度数据上采样两倍变为0.25度数据,变为最终的输出
    '''
    def __init__(self, 
                 # 上采样Upsample参数
                 in_channels=384, 
                 out_channels=69, 
                 high_resolution=(90, 180),
                 upsample_size=(180, 360),
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.high_resolution = high_resolution
        self.upsample_size = upsample_size
        # 上采样
        self.patch_expand = PatchExpand(input_resolution=high_resolution, dim=in_channels*2, norm_layer=norm_layer)
        # 线性层恢复特征维度
        self.linear = nn.Linear(in_channels//2, out_channels)
    
    def forward(self, x, shortcut_downsample):
        B, _, _ = x.shape
        # 拼接
        x = torch.cat([x, shortcut_downsample], dim=-1) # (B, 90*180, 384*2)
        # patch_expand上采样
        x = self.patch_expand(x) # (B, 180*360, 192)
        
        # 线性层恢复特征维度
        x = self.linear(x) # (B, 180*360, 69)
        # 生成最终输出
        x = x.view(B, 180, 360, 69).permute(0, 3, 1, 2).contiguous() # (B, 69, 180, 360)
        return x

class BaselineModel(nn.Module):
    def __init__(self,
                 # --- 输入数据的原始维度 ---
                 raw_in_h=180, # 原始输入高度
                 raw_in_w=360, # 原始输入宽度
                 raw_in_channels=69, # 原始输入通道数 (垂直层数)
                 final_out_channels=69, # 最终输出通道数

                 # --- 各阶段通道数 ---
                 C0=768, # DownSampleBlock 输出通道, Encoder_High 输入, Decoder_High 输出, UpsampleBlock 输入
                 C1=1536, # Encoder_High 输出通道, core 输入, Decoder_High 输入

                 # --- 各模块深度 ---
                 encoder_high_depth=10,
                 core_depth=30,
                 decoder_high_depth=10,

                 # --- Transformer 公共参数 (可以为每个模块单独设置，这里为简化设为通用或典型值) ---
                 # Encoder_High (2 degree)
                 eh_num_heads=12, eh_window_size=(6, 12), eh_shift_size=(3, 6), eh_mlp_ratio=4.,
                 # Core (2 degree)
                 core_num_heads=24, core_window_size=(5, 10), core_shift_size=(2, 5), core_mlp_ratio=4.,
                 # Decoder_High (2 degree)
                 dh_num_heads=12, dh_window_size=(6, 12), dh_shift_size=(3, 6), dh_mlp_ratio=4.,

                 # --- Dropout 和 DropPath ---
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.2,

                 # --- Core Checkpoint 参数 ---
                 high_use_checkpoints=True,
                 core_use_checkpoints=True,
                 ):

        super().__init__()
        self.raw_in_h = raw_in_h
        self.raw_in_w = raw_in_w

        # --- 确定各阶段分辨率 ---
        # 2 degree resolution (after DownSampleBlock)
        res_2_h, res_2_w = 90, 180

        # 4 degree resolution (after Encoder_High)
        res_4_h, res_4_w = res_2_h // 2, res_2_w // 2 # (45, 90)


        # --- 1. DownSampleBlock ---
        self.down_sample = DownSampleBlock(in_channels=raw_in_channels, out_channels=C0)

        # --- 2. Encoder_High ---
        self.encoder_high = Encoder_High(
            depth=encoder_high_depth,
            in_channels=C0,
            input_resolution=(res_2_h, res_2_w),
            num_heads=eh_num_heads,
            window_size=eh_window_size,
            shift_size=eh_shift_size,
            mlp_ratio=eh_mlp_ratio,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, # 使用统一的 drop rate
            out_channels=C1,
            use_checkpoints=high_use_checkpoints
        )

        

        # --- 4. Core ---
        
        self.core = Core(
            depth=core_depth,
            in_channels=C1,
            input_resolution=(res_4_h, res_4_w),
            num_heads=core_num_heads,
            window_size=core_window_size,
            shift_size=core_shift_size,
            mlp_ratio=core_mlp_ratio,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            use_checkpoints=core_use_checkpoints,
        )

        # --- 6. Decoder_High ---
        # 输入: x_dm: (B, res_1_h * res_1_w, C1), shortcut_eh: (B, res_0_5_h * res_0_5_w, C0)
        # 输出: (B, res_0_5_h * res_0_5_w, C0)
        self.decoder_high = Decoder_High(
            in_channels=C1, # Core 输出通道
            out_channels=C0, # Decoder_High 输出通道
            core_resolution=(res_4_h, res_4_w),
            depth=decoder_high_depth,
            input_resolution=(res_2_h, res_2_w), # Transformer 输入分辨率
            num_heads=dh_num_heads,
            window_size=dh_window_size,
            shift_size=dh_shift_size,
            mlp_ratio=dh_mlp_ratio,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            use_checkpoints=high_use_checkpoints
        )

        # --- 7. UpsampleBlock ---
        # 输入: x_dh: (B, res_0_5_h * res_0_5_w, C0), shortcut_ds: (B, res_0_5_h * res_0_5_w, C0)
        # 输出: (B, final_out_channels, raw_in_h, raw_in_w)
        self.upsample = UpsampleBlock(
            in_channels=C0, # Decoder_High 输出通道
            out_channels=final_out_channels,
            high_resolution=(res_2_h, res_2_w),
            upsample_size=(raw_in_h, raw_in_w) # 最终输出分辨率
        )

    def forward(self, x):
        """
        定义模型的前向传播逻辑。
        Args:
            x (torch.Tensor): 输入张量，形状为 (B, 2, raw_in_channels, raw_in_h, raw_in_w)。
        Returns:
            torch.Tensor or tuple:
                - 如果模型处于评估模式 (eval) 或 Core 未使用 MoE (或 MoE 未返回 l_aux)，
                  则返回形状为 (B, final_out_channels, raw_in_h, raw_in_w) 的输出张量。
                - 如果模型处于训练模式 (train) 且 Core 使用了 MoE 并返回了辅助损失 l_aux，
                  则返回一个元组 (output, l_aux)。
        """
        input_dtype = x.dtype # 记录输入数据类型

        # 1. DownSampleBlock
        x, shortcut_downsample = self.down_sample(x)
        # print(f"After DownSample: x={x.shape}, shortcut_ds={shortcut_downsample.shape}")

        # 2. Encoder_High
        x, shortcut_encoder_high = self.encoder_high(x)
        # print(f"After Encoder_High: x={x.shape}, shortcut_eh={shortcut_encoder_high.shape}")

        # 3. Core
        core_output = self.core(x)
        x = core_output
        # print(f"After Core: x={x.shape}, l_aux={l_aux}")

        # 4. Decoder_High
        x = self.decoder_high(x, shortcut_encoder_high)
        # print(f"After Decoder_High: x={x.shape}")

        # 5. UpsampleBlock
        x = self.upsample(x, shortcut_downsample)
        # print(f"After Upsample: x={x.shape}")

        # --- 类型检查与转换 ---
        if x.dtype != input_dtype:
             x = x.to(input_dtype)

        return x

if __name__ == "__main__":
    # 定义输入参数
    batch_size = 1
    in_d, in_c, in_h, in_w = 2, 69, 180, 360
    final_out_channels = 69 # 最终输出通道
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建一个随机输入张量
    input_tensor = torch.randn(batch_size, in_d, in_c, in_h, in_w)
    print(f"输入张量形状: {input_tensor.shape}, 数据类型: {input_tensor.dtype}")
    input_tensor = input_tensor.to(device)


    model = BaselineModel()
    # 将模型移动到指定设备
    model.to(device)

    # 前向传播
    import time
    start_time = time.time()
    output = model(input_tensor)
    end_time = time.time()
    print(f"前向传播完成，用时: {end_time - start_time:.2f}s")

    
    # 打印模型参数统计信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params/1e9:.2f}B")
    print(f"模型参数量(MB): {total_params * 4 / (1024 * 1024):.2f}MB")