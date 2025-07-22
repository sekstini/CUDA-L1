import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(OptimizedConv2d, self).__init__()
        
        # Handle kernel_size as tuple or int
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights with the same initialization as PyTorch's Conv2d
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1]
        ))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        
        # Initialize bias if needed
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias)
            
        # Optimization-related attributes
        self.register_buffer('weight_optimized', None)
        self.stream = None
        self.initialized = False
        self.last_device = None
        self.input_shape = None
        
        # Enable cuDNN benchmark mode globally for better performance
        # This allows cuDNN to select the fastest algorithm for the given input dimensions
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # For algorithm selection
        self.best_algo_found = False
        
    def _initialize_for_device(self, device, input_shape=None):
        """Initialize optimized structures for the current device"""
        # Create dedicated CUDA stream if on GPU with high priority
        if device.type == 'cuda' and self.stream is None:
            try:
                # Try to create a high-priority stream
                self.stream = torch.cuda.Stream(device=device, priority=-1)
            except:
                # Fall back to standard stream if priority not supported
                self.stream = torch.cuda.Stream(device=device)
            
        # Create optimized weight copy
        with torch.no_grad():
            # Clone and ensure contiguous memory layout
            weight_opt = self.weight.to(device).contiguous()
            
            # Store optimized weight
            self.weight_optimized = weight_opt
            
        self.initialized = True
        self.last_device = device
        
        # Store input shape for algorithm selection
        if input_shape is not None:
            self.input_shape = input_shape
    
    def forward(self, x):
        device = x.device
        input_shape = x.shape
        
        # Initialize for device if needed or if device changed
        if not self.initialized or device != self.last_device:
            self._initialize_for_device(device, input_shape)
        
        # If input shape changed, reset algorithm selection
        if self.input_shape is not None and input_shape != self.input_shape:
            self.best_algo_found = False
            self.input_shape = input_shape
            
        # Ensure input is contiguous for better memory access
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Use the optimized weight for computation
        weight_to_use = self.weight_optimized if self.weight_optimized is not None else self.weight
        
        # For CUDA tensors, use stream for potential overlap
        if device.type == 'cuda' and self.stream is not None:
            with torch.cuda.stream(self.stream):
                # If we haven't found the best algorithm yet, temporarily ensure benchmarking
                if not self.best_algo_found:
                    # Run convolution - cuDNN will benchmark and select the best algorithm
                    output = F.conv2d(
                        x, 
                        weight_to_use,
                        bias=self.bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups
                    )
                    
                    self.best_algo_found = True
                else:
                    # Use PyTorch's optimized convolution with the best algorithm
                    output = F.conv2d(
                        x, 
                        weight_to_use,
                        bias=self.bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups
                    )
                
                # Ensure computation is done before returning
                if torch.cuda.current_stream() != self.stream:
                    torch.cuda.current_stream().wait_stream(self.stream)
        else:
            # For non-CUDA tensors or if stream creation failed
            output = F.conv2d(
                x, 
                weight_to_use,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
            
        return output

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv2d = OptimizedConv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, 
            groups=groups, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.conv2d(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
width = 256
height = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization