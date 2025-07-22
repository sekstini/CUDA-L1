import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Model that performs convolution, group normalization, scaling, max pooling, and clamping.
    Optimized implementation that maintains identical functionality.
    
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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        # CUDA graph attributes
        self.static_input = None
        self.static_output = None
        self.graph = None
        self.last_input_shape = None
        
        # Cache for parameters
        self.cached_params = {}
        
        # Initialize CUDA stream and events if available
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
            self.use_cuda_graph = True
        else:
            self.use_cuda_graph = False
        
        # Create optimized forward function using TorchScript
        self._create_optimized_forward()
    
    def _create_optimized_forward(self):
        """Create an optimized version of the forward computation using TorchScript"""
        try:
            @torch.jit.script
            def optimized_forward(x: torch.Tensor, 
                                 conv_weight: torch.Tensor, 
                                 conv_bias: torch.Tensor, 
                                 gamma: torch.Tensor, 
                                 beta: torch.Tensor, 
                                 scale: torch.Tensor, 
                                 num_groups: int, 
                                 maxpool_kernel_size: int,
                                 clamp_min: float, 
                                 clamp_max: float) -> torch.Tensor:
                # Ensure input is contiguous for optimal memory access
                if not x.is_contiguous():
                    x = x.contiguous()
                
                # Apply convolution
                x = F.conv2d(x, conv_weight, conv_bias)
                
                # Apply group normalization
                x = F.group_norm(x, num_groups, gamma, beta, eps=1e-5)
                
                # Apply scaling
                x = x * scale
                
                # Apply max pooling
                x = F.max_pool2d(x, kernel_size=maxpool_kernel_size)
                
                # Apply clamping
                x = torch.clamp(x, clamp_min, clamp_max)
                
                return x
            
            self.optimized_forward = optimized_forward
            self.use_script = True
        except Exception:
            self.use_script = False
    
    def _get_params(self, x):
        """Get or create cached parameters for the optimized forward pass"""
        device = x.device
        if device not in self.cached_params:
            conv_bias = self.conv.bias if self.conv.bias is not None else torch.zeros(self.conv.out_channels, device=device)
            gamma = self.group_norm.weight if self.group_norm.weight is not None else torch.ones(self.conv.out_channels, device=device)
            beta = self.group_norm.bias if self.group_norm.bias is not None else torch.zeros(self.conv.out_channels, device=device)
            self.cached_params[device] = (conv_bias, gamma, beta)
        
        return self.cached_params[device]
    
    def _compute(self, x):
        """Compute the forward pass using the optimized implementation"""
        if self.use_script:
            # Get the necessary parameters for the optimized forward pass
            conv_bias, gamma, beta = self._get_params(x)
            
            return self.optimized_forward(
                x, 
                self.conv.weight, 
                conv_bias, 
                gamma, 
                beta, 
                self.scale,
                self.group_norm.num_groups, 
                self.maxpool_kernel_size, 
                self.clamp_min, 
                self.clamp_max
            )
        else:
            # Fallback to standard implementation
            x = self.conv(x)
            x = self.group_norm(x)
            x = x * self.scale
            x = F.max_pool2d(x, kernel_size=self.maxpool_kernel_size)
            x = torch.clamp(x, self.clamp_min, self.clamp_max)
            return x
    
    def _initialize_cuda_graph(self, x):
        """Initialize or reinitialize CUDA graph with the given input shape"""
        if not self.use_cuda_graph or not x.is_cuda:
            return False
        
        try:
            # Clean up existing resources
            self._cleanup_cuda_resources()
            
            # Record input shape
            self.last_input_shape = x.shape
            
            # Create static input tensor
            self.static_input = torch.zeros_like(x, device=x.device)
            
            # Two-phase warmup strategy
            # Phase 1: Quick initial warmup (2 iterations)
            with torch.cuda.stream(self.stream):
                for _ in range(2):
                    _ = self._compute(x)
            
            # Phase 2: More thorough warmup (3 more iterations)
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    _ = self._compute(x)
            
            # Calculate output shape
            batch_size = x.size(0)
            out_channels = self.conv.out_channels
            h_in, w_in = x.size(2), x.size(3)
            h_conv = h_in - self.conv.kernel_size[0] + 1
            w_conv = w_in - self.conv.kernel_size[1] + 1
            h_out = h_conv // self.maxpool_kernel_size
            w_out = w_conv // self.maxpool_kernel_size
            
            # Create static output tensor
            self.static_output = torch.zeros(
                (batch_size, out_channels, h_out, w_out),
                device=x.device, dtype=x.dtype
            )
            
            # Capture graph
            self.graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.stream(self.stream):
                self.static_input.copy_(x, non_blocking=True)
                with torch.cuda.graph(self.graph):
                    tmp_output = self._compute(self.static_input)
                    self.static_output.copy_(tmp_output)
            
            return True
            
        except Exception:
            # Fallback if graph capture fails
            self._cleanup_cuda_resources()
            return False
    
    def _cleanup_cuda_resources(self):
        """Clean up CUDA resources"""
        if hasattr(self, 'graph') and self.graph is not None:
            del self.graph
            self.graph = None
            
        if hasattr(self, 'static_input') and self.static_input is not None:
            del self.static_input
            self.static_input = None
            
        if hasattr(self, 'static_output') and self.static_output is not None:
            del self.static_output
            self.static_output = None
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height', width').
        """
        # Fast path: use CUDA graph if possible
        if self.use_cuda_graph and x.is_cuda:
            # Check if input shape changed or graph not initialized
            if self.graph is None or x.shape != self.last_input_shape:
                success = self._initialize_cuda_graph(x)
                
                # If initialization failed, fall back to compute
                if not success:
                    return self._compute(x)
            
            # Use graph if initialization was successful
            if self.graph is not None:
                with torch.cuda.stream(self.stream):
                    self.static_input.copy_(x, non_blocking=True)
                    self.graph.replay()
                return self.static_output
        
        # Fallback path: compute without graph
        return self._compute(x)
    
    def __del__(self):
        """Clean up CUDA resources when the module is deleted"""
        self._cleanup_cuda_resources()

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