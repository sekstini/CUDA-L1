import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with square input and asymmetric kernel, optionally with dilation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Create the standard PyTorch ConvTranspose1d layer
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        
        # Store parameters for output size calculation
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        
        # Initialize CUDA optimization variables
        self.use_cuda = torch.cuda.is_available()
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None
        self.last_input_shape = None
        
        # Enable cuDNN benchmarking for optimal algorithm selection
        if self.use_cuda:
            torch.backends.cudnn.benchmark = True
            
            # Create stream for graph capture
            self.stream = torch.cuda.Stream()
            
            # Move model to GPU and optimize with JIT
            self.conv1d_transpose = self.conv1d_transpose.cuda()
            try:
                self.conv1d_transpose = torch.jit.script(self.conv1d_transpose)
            except Exception:
                # Fallback if JIT fails
                pass
            
            # Set to evaluation mode
            self.conv1d_transpose.eval()
    
    def calculate_output_size(self, input_shape):
        """Calculate the output tensor size based on input shape and convolution parameters"""
        batch_size, _, length = input_shape
        output_length = (length - 1) * self.stride - 2 * self.padding + \
                        self.dilation * (self.kernel_size - 1) + 1
        return (batch_size, self.conv1d_transpose.out_channels, output_length)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Fall back to standard implementation if CUDA is not available
        if not self.use_cuda:
            return self.conv1d_transpose(x)
        
        # Ensure input is on the correct device and contiguous
        if x.device.type != 'cuda':
            x = x.cuda(non_blocking=True)
        
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Use dedicated stream for all operations
        with torch.cuda.stream(self.stream):
            # Check if input shape has changed or graph needs initialization
            if self.last_input_shape != x.shape or self.cuda_graph is None:
                # Clean up previous tensors if they exist
                if self.static_input is not None:
                    del self.static_input
                    del self.static_output
                
                # Update last input shape
                self.last_input_shape = x.shape
                
                # Calculate output size
                output_size = self.calculate_output_size(x.shape)
                
                # Initialize static tensors for CUDA graph
                self.static_input = torch.zeros_like(x, device='cuda')
                self.static_output = torch.zeros(output_size, device='cuda', dtype=x.dtype)
                
                # Copy input data
                self.static_input.copy_(x)
                
                # Run the convolution once before capture to warm up
                with torch.no_grad():
                    self.static_output = self.conv1d_transpose(self.static_input)
                
                # Minimal synchronization - only before graph capture
                self.stream.synchronize()
                
                # Create and capture the CUDA graph
                self.cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.cuda_graph, stream=self.stream):
                    with torch.no_grad():
                        self.static_output = self.conv1d_transpose(self.static_input)
            
            # Run the captured graph with new input data
            self.static_input.copy_(x)
            self.cuda_graph.replay()
            
            # Return the output directly - no need to clone
            return self.static_output

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 5
length = 256
stride = 1
padding = 0
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]