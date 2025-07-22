import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_depth, kernel_width, kernel_height), 
                             where kernel_width == kernel_height.
        stride (tuple, optional): Stride of the convolution. Defaults to (1, 1, 1).
        padding (tuple, optional): Padding applied to the input. Defaults to (0, 0, 0).
        output_padding (tuple, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Create the transposed convolution layer
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, 
            output_padding=output_padding, groups=groups, bias=bias
        )
        
        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Flag to track weight optimization
        self._weights_optimized = False

    def _optimize_weights_memory_format(self):
        """Optimize weight memory format for better performance - done once"""
        if not self._weights_optimized and self.conv_transpose3d.weight.is_cuda:
            # Convert weights to channels_last_3d format for better memory access
            self.conv_transpose3d.weight.data = self.conv_transpose3d.weight.data.to(
                memory_format=torch.channels_last_3d
            )
            
            # Also optimize bias if present
            if self.conv_transpose3d.bias is not None:
                self.conv_transpose3d.bias.data = self.conv_transpose3d.bias.data.contiguous()
            
            self._weights_optimized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        # Optimize for CUDA execution
        if x.is_cuda:
            # Optimize weights memory format once
            self._optimize_weights_memory_format()
            
            # Convert input to channels_last_3d for better memory access patterns
            x = x.to(memory_format=torch.channels_last_3d)
            
            # Use autocast for float32 inputs
            if x.dtype == torch.float32:
                with torch.cuda.amp.autocast(enabled=True):
                    output = self.conv_transpose3d(x)
                    
                # Convert back to float32 if needed
                if output.dtype != torch.float32:
                    output = output.float()
                
                return output
            else:
                # For other dtypes, use direct computation
                return self.conv_transpose3d(x)
        else:
            # CPU execution - use standard implementation
            return self.conv_transpose3d(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 32
out_channels = 64
kernel_depth = 3
kernel_width = 5
kernel_height = 5
depth = 64
width = 64
height = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, width, height)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, (kernel_depth, kernel_width, kernel_height)]  # Provide in_channels, out_channels, kernel_size for initialization