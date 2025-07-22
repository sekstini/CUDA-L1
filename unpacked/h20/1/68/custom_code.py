import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedConvTranspose3d(nn.Module):
    """
    Optimized implementation of 3D transposed convolution using memory format optimization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), 
                 padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1, bias=False):
        super(OptimizedConvTranspose3d, self).__init__()
        
        # Convert parameters to tuples if they are not already
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding, output_padding)
        
        # Create the standard PyTorch implementation for initialization and fallback
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, bias=bias
        )
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Store the weight and bias from the standard implementation
        self.weight = self.conv_transpose3d.weight
        self.bias = self.conv_transpose3d.bias
        
        # Optimization flags and caches
        self.use_channels_last = False
        self.benchmark_complete = False
        self.weight_cl = None
        self.weight_version = None
        
        # Create dedicated CUDA stream for operations
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Enable cuDNN benchmarking and set flags for better performance
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def _update_weight_cache(self):
        """Update the cached weight if the original weight has changed."""
        current_version = self.weight._version
        if self.weight_cl is None or self.weight_version != current_version:
            self.weight_version = current_version
            if self.use_channels_last:
                with torch.cuda.stream(self.stream):
                    self.weight_cl = self.weight.to(memory_format=torch.channels_last_3d)
    
    def _benchmark_algorithms(self, x):
        """Benchmark different algorithms and configurations to find the fastest."""
        if self.benchmark_complete:
            return
        
        # Only benchmark if input is on CUDA
        if not x.is_cuda:
            self.benchmark_complete = True
            return
        
        try:
            # Try channels_last format
            with torch.cuda.stream(self.stream):
                x_cl = x.to(memory_format=torch.channels_last_3d)
                weight_cl = self.weight.to(memory_format=torch.channels_last_3d)
                
                # Warmup with more iterations for better stability
                for _ in range(10):
                    _ = F.conv_transpose3d(
                        x_cl, weight_cl, self.bias, self.stride,
                        self.padding, self.output_padding, self.groups
                    )
                
                # Benchmark channels_last with more iterations
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                for _ in range(20):
                    _ = F.conv_transpose3d(
                        x_cl, weight_cl, self.bias, self.stride,
                        self.padding, self.output_padding, self.groups
                    )
                end.record()
                torch.cuda.synchronize()
                cl_time = start.elapsed_time(end)
                
                # Benchmark standard format
                x_std = x.contiguous()
                weight_std = self.weight.contiguous()
                
                # Warmup standard format
                for _ in range(10):
                    _ = F.conv_transpose3d(
                        x_std, weight_std, self.bias, self.stride,
                        self.padding, self.output_padding, self.groups
                    )
                
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                for _ in range(20):
                    _ = F.conv_transpose3d(
                        x_std, weight_std, self.bias, self.stride,
                        self.padding, self.output_padding, self.groups
                    )
                end.record()
                torch.cuda.synchronize()
                std_time = start.elapsed_time(end)
                
                # Use the faster format
                self.use_channels_last = cl_time < std_time
                if self.use_channels_last:
                    self.weight_cl = weight_cl
                    self.weight_version = self.weight._version
            
            self.benchmark_complete = True
        except Exception:
            # If benchmarking fails, default to standard format
            self.use_channels_last = False
            self.benchmark_complete = True
    
    def forward(self, x):
        # Check if input is on CUDA
        if x.is_cuda:
            # Benchmark algorithms if not done yet
            if not self.benchmark_complete:
                self._benchmark_algorithms(x)
            
            try:
                # Use channels_last format if it's faster
                if self.use_channels_last:
                    # Update cached weight if needed
                    self._update_weight_cache()
                    
                    # Convert input to channels_last format
                    with torch.cuda.stream(self.stream):
                        x_cl = x.to(memory_format=torch.channels_last_3d)
                        
                        # Use the optimized format for computation
                        result = F.conv_transpose3d(
                            x_cl, self.weight_cl, self.bias, self.stride,
                            self.padding, self.output_padding, self.groups
                        )
                    
                    return result
                else:
                    # Use standard format with cuDNN optimizations
                    return F.conv_transpose3d(
                        x, self.weight, self.bias, self.stride,
                        self.padding, self.output_padding, self.groups
                    )
            except Exception:
                # Fall back to standard implementation if optimization fails
                return self.conv_transpose3d(x)
        else:
            # Use standard implementation for CPU
            return self.conv_transpose3d(x)

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
        self.conv_transpose3d = OptimizedConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, 
            output_padding=output_padding, groups=groups, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, width_out, height_out).
        """
        return self.conv_transpose3d(x)

# Keeping ALL hyperparameters EXACTLY as shown in the reference implementation
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