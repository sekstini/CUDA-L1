import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies GELU, and normalizes with GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        # Standard PyTorch implementation for reference
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        # Store parameters for custom implementation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups  # Note: This is not used in the reference implementation
        self.num_groups = num_groups
        
        # For optimized implementation
        self.weight = self.conv_transpose.weight
        self.bias = self.conv_transpose.bias
        
        # Set up cuDNN benchmark mode for potentially faster convolutions
        torch.backends.cudnn.benchmark = True

    def forward(self, x):
        # Step 1: Transposed Convolution - Use PyTorch's implementation with cuDNN
        # This is already highly optimized for most cases
        x = F.conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride,
            padding=0,  # Default padding in the reference implementation
            output_padding=0,  # Default output_padding in the reference implementation
            groups=1  # Default groups in the reference implementation (not using self.groups)
        )
        
        # Step 2: GELU activation
        x = F.gelu(x)
        
        # Step 3: GroupNorm
        x = F.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.group_norm.weight,
            bias=self.group_norm.bias,
            eps=1e-5  # Default epsilon value
        )
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 32
out_channels = 64
height, width = 32, 32
kernel_size = 4
stride = 2
groups = 8
num_groups = 8

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, groups, num_groups]