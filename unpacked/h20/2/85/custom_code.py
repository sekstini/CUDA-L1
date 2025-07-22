import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs convolution, group normalization, scaling, max pooling, and clamping.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolution kernel
        num_groups (int): Number of groups for group normalization
        scale_shape (tuple): Shape of the scaling parameter
        maxpool_kernel_size (int): Size of the max pooling kernel
        clamp_min (float): Minimum value for clamping
        clamp_max (float): Maximum value for clamping
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # Initialize standard PyTorch layers
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        # Enable cuDNN benchmarking for faster convolutions
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # For CUDA graph optimization
        self.use_cuda_graph = torch.cuda.is_available()
        self.static_input = None
        self.static_output = None
        self.cuda_graph = None
        self.graph_stream = None if not torch.cuda.is_available() else torch.cuda.Stream()
        self.initialized = False
        self.input_shape = None
        
        # For TorchScript optimization
        self.use_script = torch.cuda.is_available()
        if self.use_script:
            try:
                # Disable profiling for more aggressive optimization
                if hasattr(torch._C, '_jit_set_profiling_executor'):
                    torch._C._jit_set_profiling_executor(False)
                if hasattr(torch._C, '_jit_set_profiling_mode'):
                    torch._C._jit_set_profiling_mode(False)
                
                # Script the forward implementation
                self.scripted_forward = torch.jit.script(self._forward_impl)
            except Exception:
                self.use_script = False
    
    def _forward_impl(self, x):
        """
        Optimized implementation of the forward pass
        """
        # Ensure contiguous memory layout for better performance
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Apply convolution
        x = self.conv(x)
        
        # Apply group normalization
        x = self.group_norm(x)
        
        # Apply scaling (in-place to reduce memory traffic)
        x.mul_(self.scale)
        
        # Apply max pooling
        x = self.maxpool(x)
        
        # Apply clamping (in-place to reduce memory traffic)
        x.clamp_(self.clamp_min, self.clamp_max)
        
        return x
    
    def _calculate_output_dims(self, input_shape):
        """
        Calculate output dimensions based on input shape
        """
        batch_size, _, height, width = input_shape
        # Convolution output dimensions
        conv_out_height = height - self.conv.kernel_size[0] + 1
        conv_out_width = width - self.conv.kernel_size[1] + 1
        # MaxPool output dimensions
        out_height = conv_out_height // self.maxpool.kernel_size
        out_width = conv_out_width // self.maxpool.kernel_size
        
        return (batch_size, self.conv.out_channels, out_height, out_width)
    
    def _warmup(self, x):
        """
        Perform warmup iterations to ensure CUDA kernels are compiled
        """
        with torch.no_grad():
            # Ensure kernels are compiled before graph capture
            for _ in range(3):
                if self.use_script:
                    _ = self.scripted_forward(x)
                else:
                    _ = self._forward_impl(x)
                torch.cuda.synchronize()
    
    def _initialize_cuda_graph(self, x):
        """
        Initialize CUDA graph for the given input tensor
        """
        if not x.is_cuda:
            return False
            
        try:
            # Clean up previous graph resources if they exist
            self.static_input = None
            self.static_output = None
            self.cuda_graph = None
            
            # Store input shape for future reference
            self.input_shape = x.shape
            
            # Perform operations in the dedicated stream
            with torch.cuda.stream(self.graph_stream):
                # Perform warmup to ensure CUDA kernels are compiled
                self._warmup(x)
                
                # Calculate output dimensions
                output_dims = self._calculate_output_dims(x.shape)
                
                # Initialize static tensors for CUDA graph
                self.static_input = torch.zeros_like(x, device=x.device)
                self.static_output = torch.zeros(
                    output_dims,
                    device=x.device,
                    dtype=x.dtype
                )
                
                # Capture the CUDA graph
                self.graph_stream.synchronize()
                self.cuda_graph = torch.cuda.CUDAGraph()
                
                with torch.cuda.graph(self.cuda_graph, stream=self.graph_stream):
                    self.static_input.copy_(x)
                    if self.use_script:
                        output = self.scripted_forward(self.static_input)
                    else:
                        output = self._forward_impl(self.static_input)
                    self.static_output.copy_(output)
            
            self.initialized = True
            return True
            
        except Exception:
            # If graph capture fails, disable it for future runs
            self.use_cuda_graph = False
            return False
    
    def forward(self, x):
        """
        Optimized forward pass using CUDA graph capture and TorchScript
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        # Fast path for CPU tensors (no CUDA optimization)
        if not x.is_cuda:
            return self._forward_impl(x)
            
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Use CUDA graph for static input shapes when possible
        if self.use_cuda_graph:
            # Initialize graph on first run or if input shape changes
            if not self.initialized or self.input_shape != x.shape:
                if not self._initialize_cuda_graph(x):
                    # If initialization fails, fall back to scripted forward
                    if self.use_script:
                        return self.scripted_forward(x)
                    else:
                        return self._forward_impl(x)
            
            try:
                # Copy input to static tensor and replay the graph
                with torch.cuda.stream(self.graph_stream):
                    self.static_input.copy_(x)
                    self.cuda_graph.replay()
                
                # Return the static output without unnecessary synchronization
                return self.static_output
            except Exception:
                # If replay fails, fall back to scripted forward
                if self.use_script:
                    return self.scripted_forward(x)
                else:
                    return self._forward_impl(x)
                
        # If CUDA graph is disabled but scripting is available, use it
        elif self.use_script:
            return self.scripted_forward(x)
        
        # Fallback to standard implementation
        return self._forward_impl(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
num_groups = 8
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]