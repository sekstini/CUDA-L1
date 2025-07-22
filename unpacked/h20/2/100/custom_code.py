import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedConvTranspose3dClampDiv(nn.Module):
    """
    A fused module that combines ConvTranspose3d, clamp, and division operations
    to minimize memory traffic and kernel launches
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(FusedConvTranspose3dClampDiv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.min_value = min_value
        self.divisor = divisor
        
        # Create the convolution weight and bias with optimal memory layout
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # Initialize parameters similar to PyTorch's default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Pre-allocate CUDA stream for potential operation overlap
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
    
    def forward(self, x):
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Use stream for potential performance benefits
        if self.stream is not None and x.is_cuda:
            with torch.cuda.stream(self.stream):
                # Perform transposed convolution
                output = F.conv_transpose3d(
                    x, 
                    self.weight, 
                    self.bias, 
                    stride=self.stride, 
                    padding=self.padding
                )
                
                # Apply clamp and division in-place
                output.clamp_(min=self.min_value)
                output.div_(self.divisor)
                
                # Ensure synchronization
                torch.cuda.current_stream().wait_stream(self.stream)
                return output
        else:
            # Perform transposed convolution
            output = F.conv_transpose3d(
                x, 
                self.weight, 
                self.bias, 
                stride=self.stride, 
                padding=self.padding
            )
            
            # Apply clamp and division in-place
            output.clamp_(min=self.min_value)
            output.div_(self.divisor)
            
            return output

class ModelNew(nn.Module):
    """
    An optimized model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolving kernel
        stride (int): Stride of the convolution
        padding (int): Padding added to all sides of the input
        min_value (float): Minimum value for clamping
        divisor (float): Divisor for the final division operation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        
        # Create the fused operation module
        self.fused_op = FusedConvTranspose3dClampDiv(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            min_value, 
            divisor
        )
    
    def forward(self, x):
        """
        Optimized forward pass that fuses ConvTranspose3d, clamp, and division operations
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width)
            
        Returns:
            torch.Tensor: Output tensor after transposed convolution, clamping, and division
        """
        return self.fused_op(x)

import math

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 32
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]