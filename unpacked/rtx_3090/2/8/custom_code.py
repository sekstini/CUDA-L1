import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (tuple or int): Size of the convolving kernel
        divisor (float): Divisor to apply after convolution
        pool_size (tuple or int): Size of the max pooling window
        bias_shape (tuple): Shape of the bias tensor
        sum_dim (int): Dimension to sum along
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        
        # Initialize convolution layer
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        
        # Pre-scale weights by divisor (operation fusion)
        with torch.no_grad():
            self.conv.weight.data = self.conv.weight.data / divisor
            if self.conv.bias is not None:
                self.conv.bias.data = self.conv.bias.data / divisor
        
        self.pool_size = pool_size
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        
        # Enable cuDNN benchmarking for optimal kernel selection
        torch.backends.cudnn.benchmark = True
        
        # Convert weights to channels_last format for better memory access
        self.conv = self.conv.to(memory_format=torch.channels_last_3d)
        
        # Calculate output sizes after convolution and pooling for pre-allocation
        self.conv_output_shape = None
        self.pool_output_shape = None
        
        # Create dedicated CUDA streams for better concurrency
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def forward(self, x):
        # Use CUDA stream for better concurrency if available
        if x.is_cuda and self.stream is not None:
            with torch.cuda.stream(self.stream):
                return self._forward_impl(x)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        # Convert to channels_last memory format for better performance on CUDA
        if x.is_cuda:
            x = x.to(memory_format=torch.channels_last_3d, non_blocking=True)
        
        # Apply convolution (division is already fused into the weights)
        x = self.conv(x)
        
        # Apply max pooling using functional interface with explicit stride
        x = F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)
        
        # Optimize global average pooling with direct mean calculation
        # This is more efficient than using AdaptiveAvgPool3d for reducing to (1,1,1)
        x = x.mean(dim=(2, 3, 4), keepdim=True)
        
        # Fuse bias addition and summation in one operation
        x = x + self.bias
        result = x.sum(dim=self.sum_dim)
        
        return result

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]