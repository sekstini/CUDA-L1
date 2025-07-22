import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ===== OPTIMIZATION TOGGLE SWITCHES =====
# Configuration: 2_3_4
# Generated file: 2_3_4.py
USE_CHANNELS_LAST_3D = False      # 1: Memory format optimization
USE_CUDNN_AUTOTUNING = True      # 2: cuDNN algorithm selection
USE_TF32_ACCELERATION = True      # 3: TensorFloat-32 on Ampere+ GPUs
USE_CUDA_STREAMS = True      # 4: Dedicated CUDA streams
USE_CACHING = False      # 5: Shape/format/device caching
# =========================================

class OptimizedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), 
                 padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False):
        super(OptimizedConv3d, self).__init__()
        
        # Convert scalar values to tuples if needed
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights
        weight = torch.empty(out_channels, in_channels // groups, *kernel_size)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        
        # Store weight as parameter
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
        
        # Tech 2: Enable cuDNN optimizations
        if torch.cuda.is_available() and USE_CUDNN_AUTOTUNING:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
        # Tech 3: Enable TF32 acceleration
        if torch.cuda.is_available() and USE_TF32_ACCELERATION:
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
        
        # Tech 4: Create a dedicated CUDA stream
        self.stream = torch.cuda.Stream() if (torch.cuda.is_available() and USE_CUDA_STREAMS) else None
        
        # Tech 5: Cache for tensor status
        if USE_CACHING:
            self.last_input_shape = None
            self.last_input_device = None
            self.input_needs_format = True
            self.weight_converted = False
            self.initialized = False
        else:
            self.initialized = True  # Skip initialization routine
    
    def _initialize_on_device(self, x):
        if not self.initialized and USE_CACHING:
            # Create a dummy input for cuDNN algorithm selection
            dummy_shape = (min(x.shape[0], 2), self.in_channels, 
                          min(x.shape[2], 8), min(x.shape[3], 8), min(x.shape[4], 8))
            dummy_input = torch.zeros(dummy_shape, dtype=x.dtype, device=x.device)
            
            # Tech 1: Convert dummy input to channels_last_3d
            if USE_CHANNELS_LAST_3D:
                dummy_input = dummy_input.contiguous(memory_format=torch.channels_last_3d)
            
            # Ensure weight is on the same device
            if self.weight.device != x.device:
                self.weight.data = self.weight.data.to(device=x.device)
                if self.bias is not None:
                    self.bias.data = self.bias.data.to(device=x.device)
            
            # Tech 1: Convert weight to channels_last_3d
            if x.is_cuda and USE_CHANNELS_LAST_3D:
                self.weight.data = self.weight.data.contiguous(memory_format=torch.channels_last_3d)
                self.weight_converted = True
            
            # Run a dummy forward pass to initialize cuDNN algorithms
            if USE_CUDNN_AUTOTUNING:
                with torch.no_grad():
                    F.conv3d(dummy_input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
            
            if x.is_cuda:
                torch.cuda.synchronize()
            
            self.last_input_device = x.device
            self.initialized = True
    
    def forward(self, x):
        # Initialize or handle device change
        if USE_CACHING and (not self.initialized or self.last_input_device != x.device):
            self._initialize_on_device(x)
        
        # Move weights to the same device as input if needed
        if self.weight.device != x.device:
            self.weight.data = self.weight.data.to(x.device)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(x.device)
            if USE_CACHING:
                self.weight_converted = False
                self.last_input_device = x.device
        
        if x.is_cuda:
            # Tech 5: Check if input shape has changed (caching)
            if USE_CACHING:
                if self.last_input_shape != x.shape:
                    self.last_input_shape = x.shape
                    self.input_needs_format = not x.is_contiguous(memory_format=torch.channels_last_3d)
                
                # Tech 1: Convert input to channels_last_3d if needed
                if USE_CHANNELS_LAST_3D and self.input_needs_format:
                    x = x.contiguous(memory_format=torch.channels_last_3d)
                    self.input_needs_format = False
            else:
                # Without caching, always check and convert if needed
                if USE_CHANNELS_LAST_3D and not x.is_contiguous(memory_format=torch.channels_last_3d):
                    x = x.contiguous(memory_format=torch.channels_last_3d)
            
            # Tech 1: Ensure weight is in channels_last_3d format
            if USE_CHANNELS_LAST_3D:
                if USE_CACHING:
                    if not self.weight_converted:
                        self.weight.data = self.weight.data.contiguous(memory_format=torch.channels_last_3d)
                        self.weight_converted = True
                else:
                    # Without caching, always ensure correct format
                    if not self.weight.is_contiguous(memory_format=torch.channels_last_3d):
                        self.weight.data = self.weight.data.contiguous(memory_format=torch.channels_last_3d)
            
            # Tech 4: Use the dedicated CUDA stream
            if USE_CUDA_STREAMS and self.stream is not None:
                with torch.cuda.stream(self.stream):
                    output = F.conv3d(
                        x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups
                    )
            else:
                output = F.conv3d(
                    x, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups
                )
                
            return output
        else:
            # Fallback path for CPU tensors
            if not x.is_contiguous():
                x = x.contiguous()
            
            return F.conv3d(
                x, self.weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups
            )

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with asymmetric input and kernel sizes.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), 
                 dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = OptimizedConv3d(in_channels, out_channels, kernel_size, 
                                     stride=stride, padding=padding, 
                                     dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3d(x)

# Test configuration
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel size
depth = 16
height = 256
width = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

if __name__ == "__main__":
    print("\n===== OPTIMIZATION CONFIGURATION =====")
    print(f"Configuration: 2_3_4")
    print(f"Generated file: {filename}")
    print()
    print("Optimization status:")
    for i, (opt_name, opt_desc) in enumerate([
        ("CHANNELS_LAST_3D", "Memory format optimization"),
        ("CUDNN_AUTOTUNING", "cuDNN algorithm selection"),
        ("TF32_ACCELERATION", "TensorFloat-32 on Ampere+ GPUs"),
        ("CUDA_STREAMS", "Dedicated CUDA streams"),
        ("CACHING", "Shape/format/device caching")
    ], 1):
        status = "✅ ENABLED" if str(i) in {'3', '4', '2'} else "❌ DISABLED"
        print(f"{i}. {opt_desc}: {status}")
    print("=" * 40)
    
    # Create model
    model = ModelNew(*get_init_inputs())
    
    # Test on GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        x = get_inputs()[0].cuda()
        
        # Warmup
        for _ in range(3):
            _ = model(x)
        
        # Simple timing test
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(10):
            _ = model(x)
        end.record()
        
        torch.cuda.synchronize()
        print(f"\nAverage forward pass time: {start.elapsed_time(end) / 10:.2f} ms")
