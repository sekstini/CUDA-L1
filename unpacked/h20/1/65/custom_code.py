import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Create the standard transposed convolution layer
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, bias=bias
        )
        
        # Store parameters for optimization
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Pre-compute output dimensions for the specific input size
        self.input_height = height  # Using the global height variable
        self.input_width = width    # Using the global width variable
        self.output_height = (self.input_height - 1) * stride + kernel_size[0] - 2 * padding + output_padding
        self.output_width = (self.input_width - 1) * stride + kernel_size[1] - 2 * padding + output_padding
        
        # Efficient weight caching system
        self.weight_version = None
        self.weight_cache = None
        self.bias_cache = None
        
        # Output tensor caching
        self.output_cache = {}
        
        # Performance tracking and optimization selection
        self.call_count = 0
        self.warmup_calls = 3
        self.use_stream_parallelism = True
        
        # CUDA streams for parallelism - only create if CUDA is available
        if torch.cuda.is_available():
            self.compute_stream = torch.cuda.Stream()
            self.bias_stream = torch.cuda.Stream()
    
    def _update_cached_weights(self, device, dtype):
        """Efficient weight caching with version tracking"""
        current_version = self.conv_transpose2d.weight._version
        
        if self.weight_version != current_version or self.weight_cache is None:
            # Update version
            self.weight_version = current_version
            
            # Get and prepare weights
            self.weight_cache = self.conv_transpose2d.weight.detach().to(device=device, dtype=dtype).contiguous()
            
            # Cache bias if present
            if self.conv_transpose2d.bias is not None:
                self.bias_cache = self.conv_transpose2d.bias.detach().to(device=device, dtype=dtype).contiguous()
            else:
                self.bias_cache = None
    
    def _get_cached_output_tensor(self, batch_size, device, dtype):
        """Pre-allocated output tensor caching"""
        cache_key = (batch_size, device, dtype)
        
        if cache_key not in self.output_cache:
            self.output_cache[cache_key] = torch.empty(
                batch_size, self.out_channels, self.output_height, self.output_width,
                dtype=dtype, device=device
            )
            
            # Limit cache size to prevent memory leaks
            if len(self.output_cache) > 2:
                oldest_key = next(iter(self.output_cache))
                del self.output_cache[oldest_key]
        
        return self.output_cache[cache_key]
    
    def _optimized_direct(self, x):
        """Direct computation with cached weights"""
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get cached weights
        self._update_cached_weights(x.device, x.dtype)
        
        # Direct computation with cached weights
        return F.conv_transpose2d(
            x, self.weight_cache, self.bias_cache,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )
    
    def _optimized_stream_parallel(self, x):
        """Stream parallelism with separate bias addition"""
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Get cached weights
        self._update_cached_weights(x.device, x.dtype)
        
        # Main computation in compute stream
        with torch.cuda.stream(self.compute_stream):
            result = F.conv_transpose2d(
                x, self.weight_cache, None,  # No bias in main computation
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                groups=self.groups
            )
        
        # Add bias in parallel stream if present
        if self.bias_cache is not None:
            with torch.cuda.stream(self.bias_stream):
                self.bias_stream.wait_stream(self.compute_stream)
                result.add_(self.bias_cache.view(1, -1, 1, 1))
        
        return result
    
    def _test_methods(self, x):
        """Test both methods and select the faster one"""
        # Warm up GPU
        _ = self.conv_transpose2d(x.contiguous())
        torch.cuda.synchronize()
        
        # Test direct method
        start_event1 = torch.cuda.Event(enable_timing=True)
        end_event1 = torch.cuda.Event(enable_timing=True)
        
        start_event1.record()
        result1 = self._optimized_direct(x)
        end_event1.record()
        torch.cuda.synchronize()
        
        time1 = start_event1.elapsed_time(end_event1)
        
        # Test stream parallel method
        start_event2 = torch.cuda.Event(enable_timing=True)
        end_event2 = torch.cuda.Event(enable_timing=True)
        
        start_event2.record()
        result2 = self._optimized_stream_parallel(x)
        end_event2.record()
        torch.cuda.synchronize()
        
        time2 = start_event2.elapsed_time(end_event2)
        
        # Select the faster method
        self.use_stream_parallelism = time2 < time1
        
        return result2 if self.use_stream_parallelism else result1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # For non-CUDA tensors, use direct implementation
        if not x.is_cuda:
            return self.conv_transpose2d(x)
        
        # Warmup phase: determine the best method
        if self.call_count < self.warmup_calls:
            if self.call_count == 0:
                # First call: test both methods
                result = self._test_methods(x)
            else:
                # Subsequent warmup calls: use the selected method
                result = self._optimized_stream_parallel(x) if self.use_stream_parallelism else self._optimized_direct(x)
            
            self.call_count += 1
            return result
        
        # Production phase: use the selected method
        if self.use_stream_parallelism:
            return self._optimized_stream_parallel(x)
        else:
            return self._optimized_direct(x)


# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
width = 128
height = 128

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization