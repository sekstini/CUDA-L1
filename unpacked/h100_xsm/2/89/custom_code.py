import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super(OptimizedConvTranspose3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.kernel_size = kernel_size
        
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.stride = stride
        
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        self.padding = padding
        
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding, output_padding)
        self.output_padding = output_padding
        
        # Create standard PyTorch ConvTranspose3d module
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )
        
        # Memory format optimization
        self.memory_format = torch.channels_last_3d
        
        # Pre-convert weights to optimized format
        try:
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(memory_format=self.memory_format)
        except Exception:
            pass  # Fallback if conversion fails
        
        # Cache for algorithm selection
        self.algo_cache = {}
    
    def forward(self, x):
        # Try to use channels_last memory format for better performance with cuDNN
        try:
            # Create a cache key based on input dimensions
            cache_key = (x.shape, x.device)
            
            # Check if we've already determined the best approach for this input
            if cache_key in self.algo_cache:
                use_optimized = self.algo_cache[cache_key]
            else:
                # Default to trying optimized approach
                use_optimized = True
                self.algo_cache[cache_key] = use_optimized
            
            if use_optimized:
                # Check if input is already in the desired memory format to avoid unnecessary conversions
                if not x.is_contiguous(memory_format=self.memory_format):
                    x_optimized = x.to(memory_format=self.memory_format)
                else:
                    x_optimized = x
                
                # Use the optimized convolution
                output = self.conv_transpose(x_optimized)
                
                return output
            else:
                return self.conv_transpose(x)
        except Exception:
            # If optimization fails, update cache to avoid retrying
            if cache_key in self.algo_cache:
                self.algo_cache[cache_key] = False
                
            # Fall back to standard implementation
            return self.conv_transpose(x)

class OptimizedPostProcessing(nn.Module):
    def __init__(self, channels, pool_kernel_size, pool_stride, pool_padding):
        super(OptimizedPostProcessing, self).__init__()
        self.channels = channels
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.subtract = nn.Parameter(torch.randn(channels))
        
        # Try to create an optimized JIT compiled version of the post-processing operations
        try:
            @torch.jit.script
            def fused_post_process(x, subtract_view):
                # Apply softmax across channels (dim=1)
                x = torch.softmax(x, dim=1)
                
                # Subtract across channels
                x = x - subtract_view
                
                # Apply Swish activation: x * sigmoid(x)
                x = x * torch.sigmoid(x)
                
                # Max pooling across channels
                return torch.max(x, dim=1)[0]
            
            self.fused_post_process = fused_post_process
            self.use_jit = True
        except Exception:
            self.use_jit = False
    
    def forward(self, x):
        # Apply MaxPool3d
        x = F.max_pool3d(x, kernel_size=self.pool_kernel_size, 
                         stride=self.pool_stride, padding=self.pool_padding)
        
        # Prepare subtract view
        subtract_view = self.subtract.view(1, -1, 1, 1, 1)
        
        # Apply the remaining operations with JIT if available
        if self.use_jit:
            try:
                return self.fused_post_process(x, subtract_view)
            except Exception:
                pass
        
        # Fallback to standard implementation
        x = torch.softmax(x, dim=1)
        x = x - subtract_view
        x = x * torch.sigmoid(x)
        return torch.max(x, dim=1)[0]

class ModelNew(nn.Module):
    """
    An optimized model that performs a sequence of operations:
        - ConvTranspose3d
        - MaxPool3d
        - Softmax
        - Subtract
        - Swish
        - Max
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        # Enable cuDNN benchmarking for faster operations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set up optimized convolution
        self.conv_transpose = OptimizedConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        
        # Set up optimized post-processing
        self.post_process = OptimizedPostProcessing(
            out_channels, pool_kernel_size, pool_stride, pool_padding
        )
        
        # Create CUDA stream for asynchronous execution
        self.stream = None
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        
        # Memory format
        self.memory_format = torch.channels_last_3d
    
    def forward(self, x):
        # Make input contiguous for better memory access patterns
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Try to use asynchronous execution with CUDA stream
        if self.stream is not None and x.is_cuda:
            with torch.cuda.stream(self.stream):
                # Convert to optimized memory format if needed
                if not x.is_contiguous(memory_format=self.memory_format):
                    x = x.to(memory_format=self.memory_format)
                
                # Apply ConvTranspose3d with optimized implementation
                x = self.conv_transpose(x)
                
                # Apply post-processing operations
                result = self.post_process(x)
                
                # Wait for all operations to complete
                torch.cuda.current_stream().wait_stream(self.stream)
                return result
        else:
            # Convert to optimized memory format if needed
            if x.is_cuda and not x.is_contiguous(memory_format=self.memory_format):
                x = x.to(memory_format=self.memory_format)
            
            # Apply ConvTranspose3d with optimized implementation
            x = self.conv_transpose(x)
            
            # Apply post-processing operations
            return self.post_process(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
pool_stride = 2
pool_padding = 0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding]