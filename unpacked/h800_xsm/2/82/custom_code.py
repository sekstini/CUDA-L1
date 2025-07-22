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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size
        
        # Pre-compute scaled bias for more efficient operations
        self.register_buffer('scale_factor_tensor', torch.tensor(scaling_factor, dtype=torch.float32))

    def forward(self, x):
        # Step 1: Convolution (using PyTorch's highly optimized implementation)
        x = self.conv(x)
        
        # Step 2: Apply tanh activation (in-place for memory efficiency)
        x = torch.tanh(x)
        
        # Step 3: Fused scaling and bias addition using PyTorch's optimized operations
        # This combines multiplication and addition in a single memory-efficient operation
        # torch.addcmul(input, tensor1, tensor2, *, value=1) -> input + value * tensor1 * tensor2
        # We use: bias + scaling_factor * x * 1 = bias + scaling_factor * x
        x = torch.addcmul(self.bias, x, self.scale_factor_tensor)
        
        # Step 4: Max pooling (using PyTorch's highly optimized implementation)
        x = F.max_pool2d(x, self.pool_kernel_size)
        
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