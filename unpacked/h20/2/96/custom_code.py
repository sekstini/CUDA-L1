import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed 3D convolution, multiplies by a scalar, applies max pooling, 
    global average pooling, and clamps the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        # Initialize the transposed convolution
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
        # Pre-scale the weights and bias to fuse the scaling operation
        with torch.no_grad():
            self.conv_transpose.weight.data.mul_(scale)
            if self.conv_transpose.bias is not None:
                self.conv_transpose.bias.data.mul_(scale)
        
        # Keep original parameters for reference
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1
        
        # CUDA optimization flags
        self.use_cuda_graph = torch.cuda.is_available() and hasattr(torch.cuda, 'make_graphed_callables')
        self.use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast')
        self.channels_last = torch.cuda.is_available() and hasattr(torch, 'channels_last_3d')
        
        # Graph capture state
        self.static_input = None
        self.cuda_graph_enabled = False
        self._graphed_forward = None
        self.warmup_done = False
        
        # Training state tracking
        self.was_training = self.training
        
        # Hardware capability detection
        if torch.cuda.is_available():
            self.device_capability = torch.cuda.get_device_capability()
            self.can_use_tensor_cores = self.device_capability[0] >= 7
            
            # Create dedicated CUDA stream
            self.stream = torch.cuda.Stream()
            
            # Convert weights to channels_last format if beneficial
            if self.channels_last:
                with torch.no_grad():
                    if self.conv_transpose.weight.dim() == 5:
                        self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(
                            memory_format=torch.channels_last_3d)
            
            # Pre-allocate buffers for intermediate results to avoid repeated allocations
            self.has_buffers = False

    def _allocate_buffers(self, x_shape):
        """Pre-allocate buffers for intermediate results"""
        if not self.has_buffers and torch.cuda.is_available():
            # Calculate output shape of conv_transpose
            batch, _, d, h, w = x_shape
            out_d = (d - 1) * self.conv_transpose.stride[0] - 2 * self.conv_transpose.padding[0] + self.conv_transpose.kernel_size[0]
            out_h = (h - 1) * self.conv_transpose.stride[1] - 2 * self.conv_transpose.padding[1] + self.conv_transpose.kernel_size[1]
            out_w = (w - 1) * self.conv_transpose.stride[2] - 2 * self.conv_transpose.padding[2] + self.conv_transpose.kernel_size[2]
            
            # Allocate buffer for conv output
            self.conv_buffer = torch.empty(
                (batch, self.conv_transpose.out_channels, out_d, out_h, out_w),
                dtype=torch.float32,
                device=x_shape[0].device if isinstance(x_shape, list) else x_shape.device
            )
            
            # Allocate buffer for maxpool output
            maxpool_d = out_d // self.maxpool.kernel_size
            maxpool_h = out_h // self.maxpool.kernel_size
            maxpool_w = out_w // self.maxpool.kernel_size
            self.maxpool_buffer = torch.empty(
                (batch, self.conv_transpose.out_channels, maxpool_d, maxpool_h, maxpool_w),
                dtype=torch.float32,
                device=x_shape[0].device if isinstance(x_shape, list) else x_shape.device
            )
            
            self.has_buffers = True

    def _do_warmup(self, x):
        """Perform a thorough warmup pass to ensure CUDA kernels are compiled"""
        if x.is_cuda and not self.warmup_done:
            with torch.no_grad():
                # Use the actual input tensor for warmup
                dummy = x.clone()
                
                # Convert to channels_last if beneficial
                if self.channels_last and dummy.dim() == 5:
                    dummy = dummy.to(memory_format=torch.channels_last_3d)
                
                # Run multiple iterations to ensure kernels are properly warmed up
                for _ in range(3):
                    out = self.conv_transpose(dummy)
                    
                    if self.channels_last and out.dim() == 5:
                        out = out.to(memory_format=torch.channels_last_3d)
                        
                    out = self.maxpool(out)
                    
                    if self.channels_last and out.dim() == 5:
                        out = out.to(memory_format=torch.channels_last_3d)
                        
                    out = self.global_avg_pool(out)
                    out = torch.clamp(out, min=self.clamp_min, max=self.clamp_max)
                
                # Ensure warmup is complete
                torch.cuda.synchronize()
                self.warmup_done = True

    def _setup_cuda_graph(self, x):
        """Set up CUDA graph capture for the forward pass"""
        if not self.training and not self.cuda_graph_enabled and x.is_cuda:
            try:
                # Create static input for graph capture
                self.static_input = x.clone().detach()
                
                # Ensure the static input has the right format
                if self.channels_last and self.static_input.dim() == 5:
                    self.static_input = self.static_input.to(memory_format=torch.channels_last_3d)
                
                # Create graphed version of forward pass
                def _forward(x_graph):
                    # Apply memory format optimization if available
                    if self.channels_last and x_graph.dim() == 5:
                        x_graph = x_graph.to(memory_format=torch.channels_last_3d)
                        
                    out = self.conv_transpose(x_graph)
                    
                    if self.channels_last and out.dim() == 5:
                        out = out.to(memory_format=torch.channels_last_3d)
                        
                    out = self.maxpool(out)
                    
                    if self.channels_last and out.dim() == 5:
                        out = out.to(memory_format=torch.channels_last_3d)
                        
                    out = self.global_avg_pool(out)
                    return torch.clamp(out, min=self.clamp_min, max=self.clamp_max)
                
                # Warm up before capturing
                with torch.no_grad():
                    for _ in range(3):
                        _forward(self.static_input)
                    torch.cuda.synchronize()
                
                # Now capture the graph
                self._graphed_forward = torch.cuda.make_graphed_callables(
                    _forward, (self.static_input,))
                self.cuda_graph_enabled = True
                torch.cuda.synchronize()
            except Exception as e:
                # If graph capture fails, continue with regular execution
                self.cuda_graph_enabled = False
                self._graphed_forward = None
                self.static_input = None

    def _fused_maxpool_avgpool_clamp(self, x):
        """Fused implementation of maxpool, global average pooling, and clamp"""
        # First apply maxpool
        x = self.maxpool(x)
        
        # Then apply global average pooling
        x = torch.mean(x, dim=[2, 3, 4], keepdim=True)
        
        # Finally apply clamp
        return torch.clamp(x, min=self.clamp_min, max=self.clamp_max)

    def forward(self, x):
        # Check if training state changed - if so, reset graph capture
        if self.training != self.was_training:
            self.cuda_graph_enabled = False
            self._graphed_forward = None
            self.static_input = None
            self.was_training = self.training
        
        # Ensure input is contiguous with optimal memory format
        if self.channels_last and x.dim() == 5 and x.is_cuda:
            if not x.is_contiguous(memory_format=torch.channels_last_3d):
                x = x.contiguous(memory_format=torch.channels_last_3d)
        elif not x.is_contiguous():
            x = x.contiguous()
        
        # Do warmup if needed
        if not self.warmup_done and x.is_cuda:
            self._do_warmup(x)
        
        # Check if we can use CUDA graphs for optimization
        if self.use_cuda_graph and not self.cuda_graph_enabled and not self.training and x.is_cuda:
            self._setup_cuda_graph(x)
        
        # Use CUDA graph if available and input shape matches
        if (not self.training and 
            self.cuda_graph_enabled and 
            self._graphed_forward is not None and
            x.shape == self.static_input.shape and
            x.device == self.static_input.device):
            
            # Copy input data to our static tensor
            with torch.no_grad():
                self.static_input.copy_(x)
            
            # Execute the graph
            return self._graphed_forward(self.static_input)
        
        # Standard execution path with streams and mixed precision
        if x.is_cuda:
            with torch.cuda.stream(self.stream):
                # Convert to channels_last format if beneficial
                if self.channels_last and x.dim() == 5:
                    x = x.to(memory_format=torch.channels_last_3d)
                
                # Use mixed precision if available and tensor cores are supported
                if self.use_amp and self.can_use_tensor_cores:
                    with torch.cuda.amp.autocast():
                        # Perform convolution transpose
                        x = self.conv_transpose(x)
                        
                        # Ensure output is in channels_last format
                        if self.channels_last and x.dim() == 5:
                            x = x.to(memory_format=torch.channels_last_3d)
                        
                        # Use fused implementation for the rest of the pipeline
                        return self._fused_maxpool_avgpool_clamp(x)
                else:
                    # Perform convolution transpose
                    x = self.conv_transpose(x)
                    
                    # Ensure output is in channels_last format
                    if self.channels_last and x.dim() == 5:
                        x = x.to(memory_format=torch.channels_last_3d)
                    
                    # Use fused implementation for the rest of the pipeline
                    return self._fused_maxpool_avgpool_clamp(x)
        
        # CPU fallback path
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = self.global_avg_pool(x)
        return torch.clamp(x, min=self.clamp_min, max=self.clamp_max)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale = 0.5
maxpool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size]