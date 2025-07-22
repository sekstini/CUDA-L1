import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    An optimized model that performs a convolution, applies tanh, scaling, adds a bias term, and then max-pools.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        scaling_factor (float): Scaling factor to apply
        bias_shape (tuple): Shape of the bias tensor
        pool_kernel_size (int): Size of the max pooling kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        # Initialize convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # Register scaling factor as a buffer to keep it on device
        self.register_buffer('scaling_factor', torch.tensor(scaling_factor, dtype=torch.float32))
        
        # Initialize bias parameter
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store pool kernel size
        self.pool_kernel_size = pool_kernel_size
        
        # Enable cuDNN benchmarking for faster convolution
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
    
    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        
        # Apply max pooling first to reduce the amount of data for subsequent operations
        # This is the key insight from the highest-performing attempt
        x = F.max_pool2d(x, self.pool_kernel_size)
        
        # Apply tanh activation
        x = torch.tanh(x)
        
        # Fused scaling and bias addition using addcmul
        # addcmul: out = input + value * tensor1 * tensor2
        # Here: out = bias + 1 * x * scaling_factor
        x = torch.addcmul(self.bias, x, self.scaling_factor)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 2

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]