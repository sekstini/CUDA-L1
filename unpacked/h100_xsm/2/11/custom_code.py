import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution, batch normalization, 
    tanh activation, max pooling, and group normalization.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolving kernel
        stride (int): Stride of the convolution
        padding (int): Padding added to all sides of the input
        groups (int): Number of blocked connections from input channels to output channels
        num_groups (int): Number of groups for GroupNorm
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        # CUDA graph optimization
        self._cuda_graphs = {}
        self._static_inputs = {}
        self._static_outputs = {}
        self._use_cuda_graphs = True
        self._graph_enabled = torch.cuda.is_available() and hasattr(torch.cuda, 'graph')
        
        # State tracking
        self._last_training_state = self.training
        self._weight_version = 0
        self._last_input_shape = None
        
        # Stream management
        if torch.cuda.is_available():
            try:
                # Highest priority for main execution
                self._main_stream = torch.cuda.Stream(priority=-1)
            except:
                self._main_stream = torch.cuda.Stream()
                
            self._warmup_stream = torch.cuda.Stream()
            self._warmup_event = torch.cuda.Event(enable_timing=False)
            
            # Pre-optimize weights
            self._optimize_weights()
        else:
            self._main_stream = None
            self._warmup_stream = None
            self._warmup_event = None
    
    def _optimize_weights(self):
        """Pre-optimize all weights for better performance"""
        # Optimize ConvTranspose2d weights
        if hasattr(self.conv_transpose, 'weight') and self.conv_transpose.weight is not None:
            weight = self.conv_transpose.weight.data
            if weight.dim() == 4:
                weight = weight.to(memory_format=torch.channels_last)
                self.conv_transpose.weight.data = weight
        
        # Optimize bias
        if hasattr(self.conv_transpose, 'bias') and self.conv_transpose.bias is not None:
            self.conv_transpose.bias.data = self.conv_transpose.bias.data.contiguous()
        
        # Optimize batch norm parameters
        for param_name in ['weight', 'bias', 'running_mean', 'running_var']:
            if hasattr(self.batch_norm, param_name):
                param = getattr(self.batch_norm, param_name)
                if param is not None:
                    param.data = param.data.contiguous()
        
        # Optimize group norm parameters
        if hasattr(self.group_norm, 'weight') and self.group_norm.weight is not None:
            self.group_norm.weight.data = self.group_norm.weight.data.contiguous()
        if hasattr(self.group_norm, 'bias') and self.group_norm.bias is not None:
            self.group_norm.bias.data = self.group_norm.bias.data.contiguous()

    def _create_cuda_graph(self, x):
        """Create and capture a CUDA graph for the forward pass"""
        # Create static input with optimal layout
        static_x = x.clone()
        if static_x.dim() == 4:
            static_x = static_x.to(memory_format=torch.channels_last)
        
        # Warmup with minimal synchronization
        with torch.cuda.stream(self._warmup_stream):
            for _ in range(2):
                _ = self._forward_impl(static_x)
            self._warmup_event.record(self._warmup_stream)
        
        self._warmup_event.synchronize()
        
        # Capture graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(self._main_stream):
            with torch.cuda.graph(g):
                static_y = self._forward_impl(static_x)
        
        return g, static_x, static_y

    def _ensure_optimal_layout(self, x):
        """Ensure tensor has optimal memory layout"""
        if x.dim() == 4 and x.is_cuda:
            if not x.is_contiguous(memory_format=torch.channels_last):
                return x.to(memory_format=torch.channels_last)
        elif not x.is_contiguous():
            return x.contiguous()
        return x

    def _forward_impl(self, x):
        """Implementation of the forward pass with optimized memory access"""
        # Ensure optimal layout
        x = self._ensure_optimal_layout(x)
        
        # Forward pass with optimized memory access
        x = self.conv_transpose(x)
        x = self._ensure_optimal_layout(x)
        
        x = self.batch_norm(x)
        x = self.tanh(x)
        x = self.max_pool(x)
        x = self.group_norm(x)
        
        return x

    def forward(self, x):
        """Forward pass with CUDA graph optimization when possible"""
        # Check for state changes
        training_changed = self.training != self._last_training_state
        shape_changed = self._last_input_shape != x.shape
        
        if training_changed or shape_changed:
            self._cuda_graphs.clear()
            self._static_inputs.clear()
            self._static_outputs.clear()
            self._last_training_state = self.training
            self._last_input_shape = x.shape
            self._weight_version += 1
        
        # Ensure input is contiguous
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Use CUDA graph if possible
        if (self._graph_enabled and self._use_cuda_graphs and x.is_cuda and 
            not self.training and not torch.is_grad_enabled()):
            
            # Create key for graph lookup
            device_idx = x.device.index if x.device.index is not None else 0
            key = (x.shape, device_idx, x.dtype, self._weight_version)
            
            # Create graph if needed
            if key not in self._cuda_graphs:
                try:
                    with torch.no_grad():
                        self._cuda_graphs[key], self._static_inputs[key], self._static_outputs[key] = self._create_cuda_graph(x)
                except Exception:
                    self._use_cuda_graphs = False
                    return self._forward_impl(x)
            
            # Execute graph
            with torch.cuda.stream(self._main_stream):
                self._static_inputs[key].copy_(x, non_blocking=True)
                self._cuda_graphs[key].replay()
                return self._static_outputs[key]
        
        # Fall back to regular forward
        with torch.cuda.stream(self._main_stream) if (self._main_stream is not None and x.is_cuda) else torch.no_grad():
            return self._forward_impl(x)
    
    def train(self, mode=True):
        """Override train method to track state changes"""
        if self.training != mode:
            self._weight_version += 1
        return super(ModelNew, self).train(mode)
        
    def eval(self):
        """Override eval method to track state changes"""
        if self.training:
            self._weight_version += 1
        return super(ModelNew, self).eval()

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 32
out_channels = 64
kernel_size = 4
stride = 2
padding = 1
groups = 8
num_groups = 4
height, width = 32, 32

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, num_groups]