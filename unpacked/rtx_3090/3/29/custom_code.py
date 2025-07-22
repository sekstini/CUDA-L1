import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
import math

class SpatialMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, B, H, W, C, num_heads, window_size, shift_size):
        ctx.save_for_backward(x, weight, bias)
        ctx.dims = (B, H, W, C, num_heads, window_size, shift_size)
        
        # Handle the case where input resolution is smaller than window size
        if min(H, W) <= window_size:
            shift_size = 0
            window_size = min(H, W)
        
        # Reshape input for processing
        x_reshaped = x.view(B, H, W, C)
        
        # Handle shifted window case
        if shift_size > 0:
            # Calculate padding
            P_l, P_r, P_t, P_b = window_size - shift_size, shift_size, window_size - shift_size, shift_size
            
            # Pad the input
            shifted_x = F.pad(x_reshaped, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
            _, _H, _W, _ = shifted_x.shape
            
            # Partition windows
            x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, window_size * window_size, C)  # nW*B, window_size*window_size, C
            
            # Window/Shifted-Window Spatial MLP
            x_windows_heads = x_windows.view(-1, window_size * window_size, num_heads, C // num_heads)
            x_windows_heads = x_windows_heads.transpose(1, 2)  # nW*B, nH, window_size*window_size, C//nH
            x_windows_heads = x_windows_heads.reshape(-1, num_heads * window_size * window_size, C // num_heads)
            
            # Apply spatial MLP using grouped convolution
            spatial_mlp_windows = F.conv1d(x_windows_heads, weight, bias, groups=num_heads)
            
            # Reshape back
            spatial_mlp_windows = spatial_mlp_windows.view(-1, num_heads, window_size * window_size, C // num_heads)
            spatial_mlp_windows = spatial_mlp_windows.transpose(1, 2)
            spatial_mlp_windows = spatial_mlp_windows.reshape(-1, window_size * window_size, C)
            
            # Merge windows
            spatial_mlp_windows = spatial_mlp_windows.view(-1, window_size, window_size, C)
            shifted_x = window_reverse(spatial_mlp_windows, window_size, _H, _W)  # B H' W' C
            
            # Reverse shift
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            # No shift case - simpler processing
            x_windows = window_partition(x_reshaped, window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, window_size * window_size, C)  # nW*B, window_size*window_size, C
            
            # Window/Shifted-Window Spatial MLP
            x_windows_heads = x_windows.view(-1, window_size * window_size, num_heads, C // num_heads)
            x_windows_heads = x_windows_heads.transpose(1, 2)  # nW*B, nH, window_size*window_size, C//nH
            x_windows_heads = x_windows_heads.reshape(-1, num_heads * window_size * window_size, C // num_heads)
            
            # Apply spatial MLP using grouped convolution
            spatial_mlp_windows = F.conv1d(x_windows_heads, weight, bias, groups=num_heads)
            
            # Reshape back
            spatial_mlp_windows = spatial_mlp_windows.view(-1, num_heads, window_size * window_size, C // num_heads)
            spatial_mlp_windows = spatial_mlp_windows.transpose(1, 2)
            spatial_mlp_windows = spatial_mlp_windows.reshape(-1, window_size * window_size, C)
            
            # Merge windows
            spatial_mlp_windows = spatial_mlp_windows.view(-1, window_size, window_size, C)
            x = window_reverse(spatial_mlp_windows, window_size, H, W)  # B H W C
        
        return x.view(B, H * W, C)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        B, H, W, C, num_heads, window_size, shift_size = ctx.dims
        
        # Handle the case where input resolution is smaller than window size
        if min(H, W) <= window_size:
            shift_size = 0
            window_size = min(H, W)
        
        # Initialize gradients
        grad_x = torch.zeros_like(x)
        grad_weight = torch.zeros_like(weight)
        grad_bias = None if bias is None else torch.zeros_like(bias)
        
        # Reshape grad_output for processing
        grad_output = grad_output.view(B, H, W, C)
        
        # Handle shifted window case
        if shift_size > 0:
            # Calculate padding
            P_l, P_r, P_t, P_b = window_size - shift_size, shift_size, window_size - shift_size, shift_size
            
            # Pad grad_output
            grad_shifted_x = F.pad(grad_output, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
            _, _H, _W, _ = grad_shifted_x.shape
            
            # Partition windows for grad_output
            grad_windows = window_partition(grad_shifted_x, window_size)
            grad_windows = grad_windows.view(-1, window_size * window_size, C)
            
            # Reshape for backward pass
            grad_windows_heads = grad_windows.view(-1, window_size * window_size, num_heads, C // num_heads)
            grad_windows_heads = grad_windows_heads.transpose(1, 2)
            grad_windows_heads = grad_windows_heads.reshape(-1, num_heads * window_size * window_size, C // num_heads)
            
            # Reshape x for backward
            x_reshaped = x.view(B, H, W, C)
            shifted_x = F.pad(x_reshaped, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
            x_windows = window_partition(shifted_x, window_size)
            x_windows = x_windows.view(-1, window_size * window_size, C)
            
            # Reshape for spatial MLP backward
            x_windows_heads = x_windows.view(-1, window_size * window_size, num_heads, C // num_heads)
            x_windows_heads = x_windows_heads.transpose(1, 2)
            x_windows_heads = x_windows_heads.reshape(-1, num_heads * window_size * window_size, C // num_heads)
            
            # Compute gradients
            # Gradient for input
            grad_input = F.conv_transpose1d(grad_windows_heads, weight, None, 1, 0, 1, num_heads)
            
            # Gradient for weight
            for h in range(num_heads):
                h_start = h * window_size * window_size
                h_end = (h + 1) * window_size * window_size
                
                # Extract data for this head
                x_h = x_windows_heads[:, h_start:h_end, :]
                grad_h = grad_windows_heads[:, h_start:h_end, :]
                
                # Compute gradient for this head's weight
                for i in range(window_size * window_size):
                    for j in range(window_size * window_size):
                        # Sum over batch dimension
                        grad_weight[h_start + i, j, 0] += torch.sum(
                            x_windows_heads[:, h_start + j, :] * grad_windows_heads[:, h_start + i, :]
                        )
            
            # Gradient for bias
            if bias is not None:
                grad_bias = grad_windows_heads.sum(0).sum(-1)
            
            # Reshape grad_input back
            grad_input = grad_input.view(-1, num_heads, window_size * window_size, C // num_heads)
            grad_input = grad_input.transpose(1, 2)
            grad_input = grad_input.reshape(-1, window_size * window_size, C)
            
            # Merge windows for grad_input
            grad_input = grad_input.view(-1, window_size, window_size, C)
            grad_shifted_input = window_reverse(grad_input, window_size, _H, _W)
            
            # Reverse shift for grad_input
            grad_x = grad_shifted_input[:, P_t:-P_b, P_l:-P_r, :].contiguous().view(B, H * W, C)
        else:
            # No shift case - simpler processing for backward
            grad_windows = window_partition(grad_output, window_size)
            grad_windows = grad_windows.view(-1, window_size * window_size, C)
            
            # Reshape for backward pass
            grad_windows_heads = grad_windows.view(-1, window_size * window_size, num_heads, C // num_heads)
            grad_windows_heads = grad_windows_heads.transpose(1, 2)
            grad_windows_heads = grad_windows_heads.reshape(-1, num_heads * window_size * window_size, C // num_heads)
            
            # Reshape x for backward
            x_reshaped = x.view(B, H, W, C)
            x_windows = window_partition(x_reshaped, window_size)
            x_windows = x_windows.view(-1, window_size * window_size, C)
            
            # Reshape for spatial MLP backward
            x_windows_heads = x_windows.view(-1, window_size * window_size, num_heads, C // num_heads)
            x_windows_heads = x_windows_heads.transpose(1, 2)
            x_windows_heads = x_windows_heads.reshape(-1, num_heads * window_size * window_size, C // num_heads)
            
            # Compute gradients
            # Gradient for input
            grad_input = F.conv_transpose1d(grad_windows_heads, weight, None, 1, 0, 1, num_heads)
            
            # Gradient for weight
            for h in range(num_heads):
                h_start = h * window_size * window_size
                h_end = (h + 1) * window_size * window_size
                
                # Extract data for this head
                x_h = x_windows_heads[:, h_start:h_end, :]
                grad_h = grad_windows_heads[:, h_start:h_end, :]
                
                # Compute gradient for this head's weight
                for i in range(window_size * window_size):
                    for j in range(window_size * window_size):
                        # Sum over batch dimension
                        grad_weight[h_start + i, j, 0] += torch.sum(
                            x_windows_heads[:, h_start + j, :] * grad_windows_heads[:, h_start + i, :]
                        )
            
            # Gradient for bias
            if bias is not None:
                grad_bias = grad_windows_heads.sum(0).sum(-1)
            
            # Reshape grad_input back
            grad_input = grad_input.view(-1, num_heads, window_size * window_size, C // num_heads)
            grad_input = grad_input.transpose(1, 2)
            grad_input = grad_input.reshape(-1, window_size * window_size, C)
            
            # Merge windows for grad_input
            grad_input = grad_input.view(-1, window_size, window_size, C)
            grad_x = window_reverse(grad_input, window_size, H, W).view(B, H * W, C)
        
        return grad_x, grad_weight, grad_bias, None, None, None, None, None, None, None

class OptimizedSwinMLPBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]  # P_l,P_r,P_t,P_b

        self.norm1 = norm_layer(dim)
        
        # Use group convolution to implement multi-head MLP
        self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 2,
                                     self.num_heads * self.window_size ** 2,
                                     kernel_size=1,
                                     groups=self.num_heads)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        
        # Apply optimized SwinMLP operation
        x = SpatialMLPFunction.apply(x, self.spatial_mlp.weight, self.spatial_mlp.bias, 
                                    B, H, W, C, self.num_heads, self.window_size, self.shift_size)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
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
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
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
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            OptimizedSwinMLPBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

class ModelNew(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                 patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
image_size = 224

def get_inputs():
    return [torch.randn(batch_size, 3, image_size, image_size)]

def get_init_inputs():
    return []