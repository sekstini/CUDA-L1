import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with an asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel (kernel_size x kernel_size).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Create weight parameter with the same shape as PyTorch's Conv3d
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, kernel_size, kernel_size, 1))
        
        # Initialize weights using the same method as PyTorch's Conv3d
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        
        # Create bias if needed
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        
        # Store parameters
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.out_channels = out_channels
        
        # Enhanced caching strategy
        self.weight_2d = None
        self.cached_device = None
        self.cached_output_dims = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out, depth_out).
        """
        batch_size, in_channels, height, width, depth = x.shape
        
        # Enhanced weight caching with device tracking
        if self.weight_2d is None or self.cached_device != x.device:
            self.weight_2d = self.weight.squeeze(-1)
            self.cached_device = x.device
        
        # Cache output dimensions to avoid recomputation
        input_key = (height, width)
        if input_key not in self.cached_output_dims:
            height_out = (height + 2 * self.padding - self.dilation * (self.weight.size(2) - 1) - 1) // self.stride + 1
            width_out = (width + 2 * self.padding - self.dilation * (self.weight.size(3) - 1) - 1) // self.stride + 1
            self.cached_output_dims[input_key] = (height_out, width_out)
        else:
            height_out, width_out = self.cached_output_dims[input_key]
        
        # Optimized tensor layout transformation using flatten/unflatten
        # This approach is more efficient than manual reshape operations
        x_transposed = x.transpose(1, 4)  # [batch, depth, height, width, channels]
        x_flattened = torch.flatten(x_transposed, 0, 1)  # [batch*depth, height, width, channels]
        x_reshaped = x_flattened.transpose(1, 3).transpose(2, 3)  # [batch*depth, channels, height, width]
        
        # Perform 2D convolution
        output_2d = F.conv2d(
            x_reshaped, 
            self.weight_2d, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )
        
        # Optimized output tensor reconstruction
        # Use unflatten for more efficient tensor reshaping
        output_transposed = output_2d.transpose(1, 2).transpose(2, 3)  # [batch*depth, height_out, width_out, out_channels]
        output_unflattened = torch.unflatten(output_transposed, 0, (batch_size, depth))  # [batch, depth, height_out, width_out, out_channels]
        output = output_unflattened.transpose(1, 4)  # [batch, out_channels, height_out, width_out, depth]
        
        return output

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
depth = 10

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width, depth)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization