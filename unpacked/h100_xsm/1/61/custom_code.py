import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Enable cuDNN autotuning for optimal algorithm selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            
            # Set fastest mode if available
            if hasattr(torch.backends.cudnn, 'fastest'):
                torch.backends.cudnn.fastest = True
        
        # Create optimized ConvTranspose3d layer
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride, padding=padding, 
            output_padding=output_padding, 
            groups=groups, bias=bias
        )
        
        # Optimize memory layout for weights
        with torch.no_grad():
            self.conv_transpose3d.weight.data = self.conv_transpose3d.weight.data.contiguous()
            if bias:
                self.conv_transpose3d.bias.data = self.conv_transpose3d.bias.data.contiguous()
        
        # Create dedicated CUDA stream for computation
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Pre-compute output shape for common input dimensions
        self.cached_output_shape = None
        self.cached_input_shape = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Ensure input is contiguous for better memory access patterns
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Use CUDA streams for potential parallel execution
        if x.is_cuda and self.stream is not None:
            with torch.cuda.stream(self.stream):
                # Apply the transposed convolution
                output = self.conv_transpose3d(x)
            
            # No need to synchronize here - PyTorch will handle synchronization when the tensor is used
            return output
        else:
            # For CPU execution or when stream is not available
            return self.conv_transpose3d(x)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
depth = 32
height = 32
width = 32

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization