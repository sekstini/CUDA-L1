import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with an asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create weight parameter with same initialization as nn.Conv2d
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters exactly as in nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        
        # Cache for optimized weight tensor in channels_last format
        self.register_buffer('weight_channels_last', None)
        self._weight_version = -1
        
        # Create a CUDA stream for computation
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Flag to track if we've warmed up cuDNN
        self._warmed_up = False
        self._best_algo = None
        self._best_memory_format = torch.channels_last  # Default to channels_last
        
        # Pre-convert weights to channels_last format during initialization
        with torch.no_grad():
            self.weight_channels_last = self.weight.contiguous(memory_format=torch.channels_last)
            self._weight_version = self.weight._version
        
        # JIT compile the convolution function for better performance
        try:
            self.optimized_conv = torch.jit.script(self._optimized_conv)
        except Exception:
            self.optimized_conv = self._optimized_conv
    
    def _optimized_conv(self, x, weight, bias, stride, padding, dilation, groups):
        """JIT-scriptable optimized convolution function"""
        return F.conv2d(x, weight, bias, stride, padding, dilation, groups)
    
    def _warmup_cudnn(self, x):
        """Enhanced multi-phase warmup for cuDNN algorithm selection"""
        if not self._warmed_up and x.is_cuda:
            with torch.no_grad():
                # Phase 1: Small batch initial exploration
                small_batch = x[:1].clone().contiguous(memory_format=torch.channels_last)
                for _ in range(3):
                    _ = F.conv2d(
                        small_batch,
                        self.weight_channels_last,
                        self.bias,
                        self.stride,
                        self.padding, 
                        self.dilation,
                        self.groups
                    )
                
                # Phase 2: Test multiple algorithm configurations with precise timing
                algos = [
                    {'deterministic': False, 'allow_tf32': True, 'benchmark': True},
                    {'deterministic': False, 'allow_tf32': True, 'benchmark': False},
                    {'deterministic': False, 'allow_tf32': False, 'benchmark': True}
                ]
                
                best_time = float('inf')
                best_algo = None
                
                for algo in algos:
                    # Save current settings
                    prev_deterministic = torch.backends.cudnn.deterministic
                    prev_tf32 = torch.backends.cudnn.allow_tf32 if hasattr(torch.backends.cudnn, 'allow_tf32') else None
                    prev_benchmark = torch.backends.cudnn.benchmark
                    
                    # Apply test configuration
                    torch.backends.cudnn.deterministic = algo['deterministic']
                    if prev_tf32 is not None:
                        torch.backends.cudnn.allow_tf32 = algo['allow_tf32']
                    torch.backends.cudnn.benchmark = algo['benchmark']
                    
                    # Warmup with current config
                    for _ in range(3):
                        _ = F.conv2d(
                            small_batch,
                            self.weight_channels_last,
                            self.bias,
                            self.stride,
                            self.padding, 
                            self.dilation,
                            self.groups
                        )
                    
                    # Time the configuration with more iterations for reliability
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    start_event.record()
                    for _ in range(10):
                        result = F.conv2d(
                            small_batch,
                            self.weight_channels_last,
                            self.bias,
                            self.stride,
                            self.padding, 
                            self.dilation,
                            self.groups
                        )
                    end_event.record()
                    
                    # Force result evaluation
                    _ = result.sum().item()
                    
                    torch.cuda.synchronize()
                    elapsed_time = start_event.elapsed_time(end_event)
                    
                    if elapsed_time < best_time:
                        best_time = elapsed_time
                        best_algo = algo
                    
                    # Restore previous settings
                    torch.backends.cudnn.deterministic = prev_deterministic
                    if prev_tf32 is not None:
                        torch.backends.cudnn.allow_tf32 = prev_tf32
                    torch.backends.cudnn.benchmark = prev_benchmark
                
                # Phase 3: Test different memory formats
                memory_formats = [torch.channels_last, torch.contiguous_format]
                best_format_time = float('inf')
                best_format = None
                
                # Apply best algorithm settings from Phase 2
                if best_algo is not None:
                    torch.backends.cudnn.deterministic = best_algo['deterministic']
                    if hasattr(torch.backends.cudnn, 'allow_tf32'):
                        torch.backends.cudnn.allow_tf32 = best_algo['allow_tf32']
                    torch.backends.cudnn.benchmark = best_algo['benchmark']
                    self._best_algo = best_algo
                
                for mem_format in memory_formats:
                    test_batch = x[:4].clone().contiguous(memory_format=mem_format)
                    test_weight = self.weight.contiguous(memory_format=mem_format)
                    
                    # Warmup
                    for _ in range(3):
                        _ = F.conv2d(
                            test_batch,
                            test_weight,
                            self.bias,
                            self.stride,
                            self.padding, 
                            self.dilation,
                            self.groups
                        )
                    
                    # Time the memory format
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    start_event.record()
                    for _ in range(5):
                        result = F.conv2d(
                            test_batch,
                            test_weight,
                            self.bias,
                            self.stride,
                            self.padding, 
                            self.dilation,
                            self.groups
                        )
                    end_event.record()
                    
                    # Force result evaluation
                    _ = result.sum().item()
                    
                    torch.cuda.synchronize()
                    elapsed_time = start_event.elapsed_time(end_event)
                    
                    if elapsed_time < best_format_time:
                        best_format_time = elapsed_time
                        best_format = mem_format
                
                self._best_memory_format = best_format if best_format is not None else torch.channels_last
                
                # Phase 4: Full batch final optimization with best settings
                full_batch = x.contiguous(memory_format=self._best_memory_format)
                
                # If best format is channels_last, use the pre-converted weight
                if self._best_memory_format == torch.channels_last:
                    test_weight = self.weight_channels_last
                else:
                    test_weight = self.weight.contiguous()
                
                for _ in range(3):
                    result = F.conv2d(
                        full_batch,
                        test_weight,
                        self.bias,
                        self.stride,
                        self.padding, 
                        self.dilation,
                        self.groups
                    )
                    # Force evaluation
                    _ = result.sum().item()
                
                torch.cuda.synchronize()
            
            self._warmed_up = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        if x.is_cuda:
            # Warmup cuDNN if needed
            if not self._warmed_up:
                self._warmup_cudnn(x)
            
            # Use CUDA stream if available
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    return self._forward_cuda(x)
            else:
                return self._forward_cuda(x)
        else:
            return self._forward_cpu(x)
    
    def _forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized forward implementation for CUDA tensors"""
        # Convert to best memory format if needed
        if not x.is_contiguous(memory_format=self._best_memory_format):
            x = x.contiguous(memory_format=self._best_memory_format)
        
        # Update cached weight if in training mode and weight has changed
        if self.training and self.weight._version != self._weight_version:
            with torch.no_grad():
                if self._best_memory_format == torch.channels_last:
                    self.weight_channels_last = self.weight.contiguous(memory_format=torch.channels_last)
                    weight_to_use = self.weight_channels_last
                else:
                    weight_to_use = self.weight.contiguous()
                self._weight_version = self.weight._version
        else:
            # Use the appropriate weight format
            if self._best_memory_format == torch.channels_last:
                weight_to_use = self.weight_channels_last
            else:
                weight_to_use = self.weight if self.weight.is_contiguous() else self.weight.contiguous()
        
        # Use optimized convolution
        return self.optimized_conv(
            x, 
            weight_to_use, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )
    
    def _forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
        """Forward implementation for CPU tensors"""
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Use F.conv2d directly which has less overhead than nn.Conv2d
        return F.conv2d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 128  # Asymmetric input

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization