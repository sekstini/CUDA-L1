import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store parameters for optimization
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # For CUDA graph optimization
        self.use_cuda_graph = False
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None
        
        # For stream optimization
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        
        # JIT compile the fused bias and tanh operation for better performance
        self.fused_bias_tanh = None
        if torch.cuda.is_available():
            try:
                # Define a JIT function that fuses bias subtraction and tanh
                @torch.jit.script
                def fused_bias_tanh(x, bias):
                    return torch.tanh(x - bias)
                
                self.fused_bias_tanh = fused_bias_tanh
            except:
                pass  # Fallback to standard operations if JIT fails
    
    def forward(self, x):
        # Check if we can use CUDA graph for inference
        if x.is_cuda and not self.training and self.use_cuda_graph and hasattr(torch.cuda, 'graph'):
            if self.cuda_graph is None or self.static_input.shape != x.shape:
                # Initialize or reinitialize CUDA graph
                self.static_input = torch.zeros_like(x)
                
                # Calculate output shape
                batch_size, _, height, width = x.shape
                out_height = (height - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
                out_width = (width - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding
                
                self.static_output = torch.zeros(
                    batch_size, self.out_channels, out_height, out_width,
                    device=x.device
                )
                
                # Create CUDA graph
                self.cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.cuda_graph):
                    self.static_input.copy_(x)
                    # Perform transposed convolution
                    conv_out = self.conv_transpose(self.static_input)
                    # Fused bias subtraction and tanh activation
                    if self.fused_bias_tanh is not None:
                        result = self.fused_bias_tanh(conv_out, self.bias)
                    else:
                        result = torch.tanh(conv_out - self.bias)
                    self.static_output.copy_(result)
                
            # Execute the captured graph
            self.static_input.copy_(x)
            self.cuda_graph.replay()
            return self.static_output
        
        # Optimized path for CUDA
        if x.is_cuda:
            with torch.cuda.stream(self.stream):
                # Step 1: Perform the transposed convolution
                conv_out = self.conv_transpose(x)
                
                # Step 2 & 3: Fused bias subtraction and tanh activation
                if self.fused_bias_tanh is not None:
                    return self.fused_bias_tanh(conv_out, self.bias)
                else:
                    # Use in-place subtraction to reduce memory usage when possible
                    if conv_out.is_contiguous():
                        return torch.tanh(conv_out.sub_(self.bias))
                    else:
                        return torch.tanh(conv_out - self.bias)
        
        # Standard path for CPU
        x = self.conv_transpose(x)
        x = x - self.bias
        x = torch.tanh(x)
        return x
    
    def enable_cuda_graph(self):
        """Enable CUDA graph optimization for inference"""
        if torch.cuda.is_available() and hasattr(torch.cuda, 'graph'):
            self.use_cuda_graph = True
            self.eval()  # Set to evaluation mode

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 32
out_channels = 16
height, width = 16, 16
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]