import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized implementation that performs a convolution, applies ReLU, and adds a bias term.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        bias_shape (tuple): Shape of the bias tensor
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Ensure parameters are contiguous for better memory access
        with torch.no_grad():
            self.conv.weight.data = self.conv.weight.data.contiguous()
            if self.conv.bias is not None:
                self.conv.bias.data = self.conv.bias.data.contiguous()
            self.bias.data = self.bias.data.contiguous()
        
        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'matmul.allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
        
        # CUDA graph related attributes
        self.graph = None
        self.static_input = None
        self.static_output = None
        self.prev_input_shape = None
        self.graph_stream = None
        
        # JIT compilation for operation fusion
        try:
            @torch.jit.script
            def optimized_forward(x, weight, bias_conv, bias_add):
                # Fused convolution, ReLU, and bias addition
                out = F.conv2d(x, weight, bias_conv)
                out.relu_()  # In-place ReLU
                out.add_(bias_add)  # In-place bias addition
                return out
            
            self.jit_forward = optimized_forward
            self.use_jit = True
            
            # Pre-warm the JIT function with a dummy input if CUDA is available
            if torch.cuda.is_available():
                dummy_input = torch.zeros(1, in_channels, 32, 32, device='cuda')
                dummy_weight = self.conv.weight.to('cuda')
                dummy_bias_conv = self.conv.bias.to('cuda') if self.conv.bias is not None else None
                dummy_bias_add = self.bias.to('cuda')
                with torch.no_grad():
                    self.jit_forward(dummy_input, dummy_weight, dummy_bias_conv, dummy_bias_add)
                    torch.cuda.synchronize()
        except Exception:
            self.use_jit = False
    
    def _create_cuda_graph(self, x):
        """
        Create and capture a CUDA graph for the forward pass
        
        Args:
            x (torch.Tensor): Input tensor with the shape to optimize for
        """
        # Only create graph if input is on CUDA
        if not x.is_cuda:
            return False
        
        try:
            # Create a dedicated stream for graph capture and replay
            if self.graph_stream is None:
                self.graph_stream = torch.cuda.Stream()
            
            # Create static input and output tensors
            self.static_input = torch.zeros_like(x)
            output_shape = (x.shape[0], self.conv.out_channels, 
                           x.shape[2] - self.conv.kernel_size[0] + 1, 
                           x.shape[3] - self.conv.kernel_size[1] + 1)
            self.static_output = torch.zeros(output_shape, device=x.device)
            
            # Record the input shape we're optimizing for
            self.prev_input_shape = x.shape
            
            # Warm up the operations to ensure cuDNN selects the optimal algorithm
            with torch.no_grad():
                for _ in range(5):  # 5 warmup iterations (balanced for performance)
                    if self.use_jit:
                        _ = self.jit_forward(x, self.conv.weight, self.conv.bias, self.bias)
                    else:
                        _ = F.conv2d(x, self.conv.weight, self.conv.bias)
                        _.relu_()
                        _.add_(self.bias)
                
                # Ensure warmup is complete
                torch.cuda.synchronize()
            
            # Capture the graph
            with torch.cuda.stream(self.graph_stream):
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    if self.use_jit:
                        result = self.jit_forward(
                            self.static_input, 
                            self.conv.weight, 
                            self.conv.bias, 
                            self.bias
                        )
                    else:
                        result = F.conv2d(self.static_input, self.conv.weight, self.conv.bias)
                        result.relu_()
                        result.add_(self.bias)
                    
                    # Copy to static output if needed
                    if result is not self.static_output:
                        self.static_output.copy_(result)
            
            return True
        except Exception:
            # If graph capture fails, fall back to regular execution
            self.graph = None
            self.static_input = None
            self.static_output = None
            return False
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Ensure input is contiguous for better memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Try to use CUDA graph if on GPU and input shape is consistent
        if x.is_cuda:
            # Check if we need to create or recreate the graph
            if self.graph is None or self.prev_input_shape != x.shape:
                success = self._create_cuda_graph(x)
            
            # If we have a valid graph, use it
            if self.graph is not None:
                try:
                    with torch.cuda.stream(self.graph_stream):
                        # Copy input data to static tensor
                        self.static_input.copy_(x)
                        # Replay the graph
                        self.graph.replay()
                    # Return the output without cloning to avoid extra memory operations
                    return self.static_output
                except Exception:
                    # If graph replay fails, fall back to regular execution
                    self.graph = None
        
        # If CUDA graph isn't available or failed, use JIT or regular execution
        if self.use_jit and x.is_cuda:
            try:
                return self.jit_forward(
                    x, 
                    self.conv.weight, 
                    self.conv.bias, 
                    self.bias
                )
            except Exception:
                # Fall back to non-JIT version if there's an error
                self.use_jit = False
        
        # Standard PyTorch implementation as final fallback
        out = F.conv2d(x, self.conv.weight, self.conv.bias)
        out.relu_()  # In-place ReLU
        out.add_(self.bias)  # In-place bias addition
        
        return out

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, bias_shape]