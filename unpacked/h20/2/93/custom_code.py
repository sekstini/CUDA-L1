import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        
        # Enable cuDNN benchmarking for faster convolution algorithm selection
        torch.backends.cudnn.benchmark = True
        
        # Allow non-deterministic algorithms for better performance
        torch.backends.cudnn.deterministic = False
        
        # Enable TF32 for faster computation on Ampere GPUs if available
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
        
        # Pre-convert weights to channels_last format if on CUDA
        if torch.cuda.is_available():
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(memory_format=torch.channels_last)
        
        # Create optimized forward implementation using TorchScript
        try:
            # Define optimized post-convolution operations
            @torch.jit.script
            def optimized_post_conv(x, add_value: float, multiply_value: float):
                # Fuse add and min operations using clamp
                x = torch.clamp(x + add_value, max=0.0)
                # Apply GELU and multiply in one sequence
                return F.gelu(x) * multiply_value
            
            self.optimized_post_conv = optimized_post_conv
            
            # Try to optimize the entire forward pass if possible
            @torch.jit.script
            def optimized_forward(module_conv, x, add_value: float, multiply_value: float):
                x = module_conv(x)
                x = torch.clamp(x + add_value, max=0.0)
                return F.gelu(x) * multiply_value
            
            self.optimized_forward = optimized_forward
        except Exception:
            self.optimized_post_conv = None
            self.optimized_forward = None

    def forward(self, x):
        # Try using channels_last memory format for better performance on 4D tensors
        if x.is_cuda and x.dim() == 4:
            try:
                # Convert input to channels_last format for better memory access patterns
                x = x.to(memory_format=torch.channels_last)
                
                # Try to use fully optimized forward pass if available
                if self.optimized_forward is not None:
                    return self.optimized_forward(self.conv_transpose, x, self.add_value, self.multiply_value)
                
                # Apply transposed convolution
                x = self.conv_transpose(x)
                
                # Use optimized post-conv operations if available
                if self.optimized_post_conv is not None:
                    return self.optimized_post_conv(x, self.add_value, self.multiply_value)
                
                # Otherwise use optimized sequence of operations
                x = torch.clamp(x + self.add_value, max=0.0)
                return F.gelu(x) * self.multiply_value
                
            except Exception:
                # Fall back to standard implementation if channels_last fails
                pass
        
        # Standard implementation fallback
        x = self.conv_transpose(x)
        x = torch.clamp(x + self.add_value, max=0.0)
        return F.gelu(x) * self.multiply_value

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 32
out_channels = 16
height, width = 32, 32
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]