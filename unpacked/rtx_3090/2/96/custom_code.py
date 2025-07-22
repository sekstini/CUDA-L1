import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OptimizedConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, scale=1.0):
        super(OptimizedConvTranspose3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        
        # Create weight parameter with correct shape for transposed convolution
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
        
        # Pre-scale the weights and bias to fuse operations
        with torch.no_grad():
            self.weight.data *= scale
            self.bias.data *= scale
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # Use PyTorch's built-in function with pre-scaled weights
        return F.conv_transpose3d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding
        )

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        
        # Use optimized ConvTranspose3d with pre-scaled weights
        self.conv_transpose = OptimizedConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, scale=scale
        )
        
        # Use PyTorch's native MaxPool3d
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        
        # Use PyTorch's native AdaptiveAvgPool3d
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.clamp_min = 0
        self.clamp_max = 1
        
        # Enable cuDNN benchmarking to find the best algorithm
        torch.backends.cudnn.benchmark = True
        
        # Store the memory format for optimized data layout
        self.memory_format = torch.channels_last_3d if torch.cuda.is_available() else torch.contiguous_format
    
    def forward(self, x):
        # Optimize memory layout if on GPU
        if x.is_cuda and not x.is_contiguous(memory_format=self.memory_format):
            x = x.contiguous(memory_format=self.memory_format)
            
        # ConvTranspose3d with pre-scaled weights
        x = self.conv_transpose(x)
        
        # Ensure contiguity between operations for better memory access
        if not x.is_contiguous(memory_format=self.memory_format):
            x = x.contiguous(memory_format=self.memory_format)
            
        # MaxPool3d
        x = self.maxpool(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        
        # Clamp values
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale = 0.5
maxpool_kernel_size = 2

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size]