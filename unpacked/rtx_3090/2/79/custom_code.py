import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    An optimized implementation of the 3D convolutional model
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        multiplier_shape (tuple): Shape of the multiplier tensor
        clamp_min (float): Minimum value for clamping
        clamp_max (float): Maximum value for clamping
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        # Optimize memory layout of parameters
        self._optimize_memory_layout()
        
        # Pre-compute some values for efficiency
        self.out_channels = out_channels
    
    def _optimize_memory_layout(self):
        """Ensure all parameters are in optimal memory layout"""
        # Ensure conv weights are contiguous in channels_last_3d format
        self.conv.weight.data = self.conv.weight.data.contiguous(memory_format=torch.channels_last_3d)
        if self.conv.bias is not None:
            self.conv.bias.data = self.conv.bias.data.contiguous()
        
        # Ensure multiplier is contiguous
        self.multiplier.data = self.multiplier.data.contiguous()
    
    def forward(self, x):
        # Convert to channels_last_3d memory format for better performance on modern GPUs
        x = x.contiguous(memory_format=torch.channels_last_3d)
        
        # Convolution
        x = self.conv(x)
        
        # First multiplication - fused with normalization preparation
        x = x * self.multiplier
        
        # Instance normalization
        x = self.instance_norm(x)
        
        # Fused clamping and second multiplication to reduce memory traffic
        x = torch.clamp(x, self.clamp_min, self.clamp_max) * self.multiplier
        
        # Max operation - use more efficient implementation
        x = torch.amax(x, dim=1)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1, 1)
clamp_min = -1.0
clamp_max = 1.0

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max]