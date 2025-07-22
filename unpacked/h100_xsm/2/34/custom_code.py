import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized implementation of a model that performs a 3D transposed convolution,
    layer normalization, GELU activation, and scaling.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): Zero-padding added to both sides of the input
        bias (bool): If True, adds a learnable bias to the output
        eps (float): A value added to the denominator for numerical stability
        scaling_factor (float): Scaling factor to apply
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        # Apply transposed convolution
        x = self.conv_transpose(x)
        
        # Store original shape for efficient reshaping
        orig_shape = x.shape
        batch_size, channels = orig_shape[0], orig_shape[1]
        
        # Reshape to 2D for LayerNorm (more efficient than view(-1, channels))
        x = x.reshape(-1, channels)
        
        # Apply LayerNorm, GELU, and scaling in sequence
        # Using native PyTorch operations for maximum efficiency
        x = self.layer_norm(x)
        x = F.gelu(x)
        
        # Apply scaling factor efficiently
        if self.scaling_factor != 1.0:
            x = x * self.scaling_factor
        
        # Reshape back to original 5D format
        return x.reshape(orig_shape)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 4
stride = 2
padding = 1
bias = True
eps = 1e-5
scaling_factor = 1.0

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]