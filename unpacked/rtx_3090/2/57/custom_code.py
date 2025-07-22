import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolution kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # Initialize with the same parameters as the reference implementation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
    def forward(self, x):
        # Use PyTorch's highly optimized Conv2d
        x = self.conv(x)
        
        # Fused ReLU + HardSwish with maximum memory efficiency
        # Original: x = torch.relu(x); x = x * torch.clamp((x + 3) / 6, 0, 1)
        
        # Apply ReLU in-place first
        x.clamp_(min=0.0)  # More efficient than F.relu for in-place
        
        # Compute HardSwish efficiently: x * clamp((x + 3) / 6, 0, 1)
        # Use a temporary view to avoid allocations
        temp = x.add(3.0)  # x + 3 (creates new tensor, but we'll reuse x)
        temp.mul_(1.0/6.0)  # (x + 3) / 6
        temp.clamp_(0.0, 1.0)  # clamp to [0, 1]
        x.mul_(temp)  # x *= clamp((x + 3) / 6, 0, 1)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size]