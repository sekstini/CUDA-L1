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
        scaling_factor (float): Scaling factor to apply after tanh
        bias_shape (tuple): Shape of the bias tensor
        pool_kernel_size (int): Size of the max pooling kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        
        # Initialize convolution layer with same parameters as reference
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # Store parameters
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size
        
        # Enable cuDNN benchmarking for optimal convolution algorithms
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Pre-compute scaling factor as a tensor for efficient GPU operations
        self.register_buffer('scale_tensor', torch.tensor(scaling_factor, dtype=torch.float32))
    
    def forward(self, x):
        # Ensure input tensor is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Apply convolution (PyTorch's optimized implementation)
        x = self.conv(x)
        
        # Apply tanh activation in-place to avoid intermediate tensor
        x = torch.tanh_(x)
        
        # Fuse scaling and bias addition in a single operation
        # This is more efficient than separate multiply and add operations
        x = torch.addcmul(self.bias, x, self.scale_tensor)
        
        # Apply max pooling with optimized parameters for 2x2 pooling
        x = F.max_pool2d(x, kernel_size=self.pool_kernel_size, stride=self.pool_kernel_size)
        
        return x

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]