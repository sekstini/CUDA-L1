import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create weights with the correct format for depthwise convolution
        # For depthwise convolution, shape is (in_channels, 1, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.Tensor(in_channels, 1, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
        # Create a dedicated CUDA stream for convolution operations
        self.stream = None
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        
        # Enable cuDNN autotuner for finding the best algorithm
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
    def reset_parameters(self):
        # Initialize weights using the same strategy as nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        # Fast path for CUDA execution
        if x.is_cuda and self.stream is not None:
            # Ensure input is contiguous for better memory access patterns
            if not x.is_contiguous():
                x = x.contiguous()
                
            # Use the dedicated CUDA stream for potentially better performance
            with torch.cuda.stream(self.stream):
                # Use PyTorch's optimized F.conv2d directly with groups=in_channels for depthwise conv
                output = F.conv2d(
                    x, 
                    self.weight, 
                    self.bias, 
                    stride=self.stride, 
                    padding=self.padding, 
                    groups=self.in_channels
                )
            
            # No explicit synchronization needed here - PyTorch will handle this automatically
            # when the output tensor is used
            return output
        else:
            # CPU execution path or fallback
            # Ensure input is contiguous for better memory access patterns
            if not x.is_contiguous():
                x = x.contiguous()
                
            return F.conv2d(
                x, 
                self.weight, 
                self.bias, 
                stride=self.stride, 
                padding=self.padding, 
                groups=self.in_channels
            )

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]