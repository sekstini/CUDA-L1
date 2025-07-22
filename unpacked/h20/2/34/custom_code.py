import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D transposed convolution, layer normalization, GELU activation, and scaling.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): Padding added to all three sides of the input
        bias (bool): If True, adds a learnable bias to the output
        eps (float): A value added to the denominator for numerical stability in LayerNorm
        scaling_factor (float): Scaling factor to apply to the output
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
        # Apply ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Get the dimensions
        batch_size, channels, depth, height, width = x.shape
        
        # Ensure the tensor is contiguous for better memory access
        x = x.contiguous()
        
        # Reshape to [batch_size * depth * height * width, channels] for LayerNorm
        x_reshaped = x.view(-1, channels)
        
        # Apply LayerNorm
        x_reshaped = self.layer_norm(x_reshaped)
        
        # Apply GELU activation
        x_reshaped = F.gelu(x_reshaped)
        
        # Apply scaling
        if self.scaling_factor != 1.0:
            x_reshaped.mul_(self.scaling_factor)
        
        # Reshape back to original 5D format
        return x_reshaped.view(batch_size, channels, depth, height, width)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
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
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]