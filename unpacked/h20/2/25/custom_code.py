import torch
import torch.nn as nn

class MinTanhTanhModule(torch.nn.Module):
    """JIT-compilable module for min + double tanh operations"""
    def forward(self, x):
        # Fuse operations to minimize intermediate memory allocations
        return torch.tanh(torch.tanh(torch.min(x, dim=1, keepdim=True)[0]))

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # Use PyTorch's optimized Conv2d implementation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # Create and JIT compile the min-tanh-tanh module
        self.min_tanh_tanh = MinTanhTanhModule()
        if torch.cuda.is_available():
            try:
                # Disable profiling for more aggressive optimization during compilation
                torch._C._jit_set_profiling_executor(False)
                torch._C._jit_set_profiling_mode(False)
                self.min_tanh_tanh = torch.jit.script(self.min_tanh_tanh)
                # Re-enable profiling after compilation
                torch._C._jit_set_profiling_executor(True)
                torch._C._jit_set_profiling_mode(True)
            except Exception:
                pass  # Fallback to non-JIT version if compilation fails
        
        # CUDA graph capture state
        self._graph_data = None
        self._is_warmed_up = False
        
        # Create a dedicated CUDA stream for better overlapping
        self._stream = None
        if torch.cuda.is_available():
            try:
                self._stream = torch.cuda.Stream()
            except Exception:
                self._stream = None
    
    def _warmup(self):
        """Perform two-phase warmup to optimize subsequent executions"""
        if self._is_warmed_up or not torch.cuda.is_available():
            return
        
        try:
            # Create sample input for warmup
            sample_input = torch.zeros(batch_size, in_channels, height, width, 
                                      device='cuda', dtype=torch.float32)
            
            # Two-phase warmup for optimal GPU state
            with torch.no_grad():
                # Phase 1: Warmup with zeros
                for _ in range(2):
                    conv_out = self.conv(sample_input)
                    _ = self.min_tanh_tanh(conv_out)
                
                # Phase 2: Warmup with values in tanh's sensitive range
                sample_input.uniform_(-1, 1)  # Values in tanh's sensitive range
                for _ in range(2):
                    conv_out = self.conv(sample_input)
                    _ = self.min_tanh_tanh(conv_out)
                
                # Reset to zeros for graph capture
                sample_input.zero_()
            
            torch.cuda.synchronize()
            self._is_warmed_up = True
        except Exception:
            # Silently continue if warmup fails
            pass
    
    def _initialize_cuda_graph(self, x):
        """Initialize CUDA graph for faster repeated execution"""
        if not torch.cuda.is_available() or not x.is_cuda:
            return None
            
        try:
            # Ensure GPU is properly warmed up before graph capture
            if not self._is_warmed_up:
                self._warmup()
            
            # Create static input tensor for graph capture with optimal memory alignment
            static_input = torch.zeros_like(x, memory_format=torch.contiguous_format)
            
            # Ensure GPU synchronization before graph capture
            torch.cuda.synchronize()
                
            # Capture the graph
            graph = torch.cuda.CUDAGraph()
            
            # Use stream if available for better overlapping
            if self._stream is not None:
                with torch.cuda.stream(self._stream):
                    with torch.cuda.graph(graph):
                        conv_out = self.conv(static_input)
                        static_output = self.min_tanh_tanh(conv_out)
            else:
                with torch.cuda.graph(graph):
                    conv_out = self.conv(static_input)
                    static_output = self.min_tanh_tanh(conv_out)
                
            return {
                'graph': graph,
                'static_input': static_input,
                'static_output': static_output
            }
        except Exception:
            # Fall back to normal execution if CUDA graphs fail
            return None
    
    def forward(self, x):
        """
        Optimized forward pass with CUDA graph acceleration
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor after convolution, min operation, and double tanh
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Fast path: use CUDA graph if available and input is on CUDA
        if x.is_cuda:
            # Initialize graph on first run or if input shape changes
            if self._graph_data is None or self._graph_data['static_input'].shape != x.shape:
                self._graph_data = self._initialize_cuda_graph(x)
            
            # Use the graph if available
            if self._graph_data is not None:
                # Efficient tensor copy with non-blocking option
                self._graph_data['static_input'].copy_(x, non_blocking=True)
                
                # Use stream if available
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        self._graph_data['graph'].replay()
                else:
                    self._graph_data['graph'].replay()
                
                # Return the output directly without cloning
                return self._graph_data['static_output']
        
        # Fallback path with JIT compilation
        conv_out = self.conv(x)
        return self.min_tanh_tanh(conv_out)
    
    def __del__(self):
        """Clean up CUDA resources"""
        self._graph_data = None
        
        # Clean up CUDA stream
        if hasattr(self, '_stream') and self._stream is not None:
            try:
                del self._stream
            except Exception:
                pass

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size]