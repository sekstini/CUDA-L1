import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with asymmetric input and square kernel.
    Supports padding, striding, and dilation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters directly as instance variables for fastest access
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create a temporary ConvTranspose1d to get properly initialized weights
        temp_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, 
                                      stride=stride, padding=padding, 
                                      dilation=dilation, bias=bias)
        
        # Store weights with optimal memory layout
        self.weight = nn.Parameter(temp_conv.weight.data)
        
        # Conditional bias initialization
        if bias:
            self.bias = nn.Parameter(temp_conv.bias.data)
        else:
            self.bias = None
        
        # Delete temporary layer to free memory
        del temp_conv
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Ultra-minimal forward pass with direct functional call
        return F.conv_transpose1d(
            x, 
            self.weight, 
            self.bias,
            self.stride,
            self.padding,
            0,  # output_padding
            1,  # groups
            self.dilation
        )

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
length = 128
stride = 2
padding = 1
dilation = 2

def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]