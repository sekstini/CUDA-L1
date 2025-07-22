import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies HardSwish, and then ReLU.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolution kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # Optimization flags
        self.use_cuda_graph = torch.cuda.is_available()
        self.use_jit = torch.cuda.is_available()
        self.use_channels_last = torch.cuda.is_available()
        
        # CUDA graph related variables
        self.graph_captured = False
        self.static_input = None
        self.static_output = None
        self.cuda_graph = None
        self.input_shape = None
        
        # Warmup status
        self.warmup_done = False
        
        # Create optimized forward function using TorchScript
        if self.use_jit:
            try:
                @torch.jit.script
                def optimized_forward(x, weight, bias):
                    # Perform convolution
                    x = F.conv2d(x, weight, bias)
                    
                    # Apply hardswish using direct formula for better fusion
                    # hardswish(x) = x * min(max(0, x + 3), 6) / 6
                    x_plus_3 = x + 3
                    clamped = torch.clamp(x_plus_3, 0, 6)
                    x = x * (clamped / 6)
                    
                    # Apply ReLU
                    x = torch.relu(x)
                    return x
                
                self.optimized_forward = optimized_forward
            except Exception:
                self.use_jit = False
        
        # Create optimized channels_last forward function
        if self.use_channels_last:
            try:
                @torch.jit.script
                def optimized_channels_last_forward(x, weight, bias):
                    # Convert to channels_last format
                    x = x.to(memory_format=torch.channels_last)
                    weight = weight.to(memory_format=torch.channels_last)
                    
                    # Perform convolution
                    x = F.conv2d(x, weight, bias)
                    
                    # Apply hardswish using direct formula
                    x_plus_3 = x + 3
                    clamped = torch.clamp(x_plus_3, 0, 6)
                    x = x * (clamped / 6)
                    
                    # Apply ReLU
                    x = torch.relu(x)
                    return x
                
                self.optimized_channels_last_forward = optimized_channels_last_forward
                
                # Pre-convert weights to channels_last
                self.conv.weight.data = self.conv.weight.data.to(memory_format=torch.channels_last)
                
                # Test if channels_last is supported
                dummy = torch.zeros(1, 1, 1, 1).to(memory_format=torch.channels_last)
            except Exception:
                self.use_channels_last = False
    
    def _warmup(self, x):
        """Perform warmup iterations to ensure JIT compilation is complete"""
        if not self.warmup_done and x.is_cuda:
            with torch.no_grad():
                # Warmup JIT path
                if self.use_jit:
                    for _ in range(5):
                        _ = self.optimized_forward(x, self.conv.weight, self.conv.bias)
                
                # Warmup channels_last path
                if self.use_channels_last:
                    x_cl = x.to(memory_format=torch.channels_last)
                    w_cl = self.conv.weight.to(memory_format=torch.channels_last)
                    for _ in range(5):
                        _ = self.optimized_channels_last_forward(x_cl, w_cl, self.conv.bias)
                
                # Warmup standard path
                for _ in range(5):
                    _ = self.conv(x)
                    _ = F.hardswish(_)
                    _ = F.relu(_)
            
            torch.cuda.synchronize()
            self.warmup_done = True
    
    def _try_capture_cuda_graph(self, x):
        """Try to capture a CUDA graph for the current input shape"""
        try:
            # Create static input and output tensors
            self.static_input = torch.zeros_like(x, device=x.device)
            self.input_shape = x.shape
            
            # Copy input data to static input
            self.static_input.copy_(x)
            
            # Capture the graph
            self.cuda_graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(self.cuda_graph):
                if self.use_channels_last:
                    self.static_output = self.optimized_channels_last_forward(
                        self.static_input, self.conv.weight, self.conv.bias
                    )
                elif self.use_jit:
                    self.static_output = self.optimized_forward(
                        self.static_input, self.conv.weight, self.conv.bias
                    )
                else:
                    self.static_output = F.relu(F.hardswish(self.conv(self.static_input)))
            
            self.graph_captured = True
            return True
        except Exception:
            self.graph_captured = False
            self.static_input = None
            self.static_output = None
            self.cuda_graph = None
            return False
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Perform warmup if needed
        if not self.warmup_done and x.is_cuda:
            self._warmup(x)
        
        # Use CUDA graph if available and input shape is static
        if self.use_cuda_graph and x.is_cuda:
            # Check if we can use the captured graph
            can_use_graph = (self.graph_captured and 
                           self.static_input is not None and
                           x.shape == self.input_shape and
                           x.device == self.static_input.device and
                           x.dtype == self.static_input.dtype)
            
            if can_use_graph:
                # Copy input data to our static input tensor
                self.static_input.copy_(x)
                # Replay the CUDA graph
                self.cuda_graph.replay()
                # Return the output
                return self.static_output
            
            # If we can't use existing graph, try to capture a new one
            elif self._try_capture_cuda_graph(x):
                # Replay the newly captured graph
                self.cuda_graph.replay()
                return self.static_output
            else:
                # If graph capture fails, fall back to regular execution
                self.use_cuda_graph = False
        
        # Try channels_last with JIT
        if x.is_cuda and self.use_channels_last:
            try:
                return self.optimized_channels_last_forward(x, self.conv.weight, self.conv.bias)
            except Exception:
                pass
        
        # Try JIT only
        if x.is_cuda and self.use_jit:
            try:
                return self.optimized_forward(x, self.conv.weight, self.conv.bias)
            except Exception:
                pass
        
        # Standard implementation (fallback)
        x = self.conv(x)
        x = F.hardswish(x)
        x = F.relu(x)
        return x


# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]