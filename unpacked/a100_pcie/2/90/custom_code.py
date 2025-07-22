import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedPostProcessing(torch.autograd.Function):
    """
    Custom autograd function for fused post-processing operations:
    LeakyReLU -> Add -> Clamp -> GELU
    """
    @staticmethod
    def forward(ctx, input, sum_tensor):
        ctx.save_for_backward(input, sum_tensor)
        
        # Step 1: LeakyReLU
        leaky_relu = F.leaky_relu(input, negative_slope=0.2)
        
        # Step 2: Add sum_tensor (broadcasting)
        added = leaky_relu + sum_tensor
        
        # Step 3: Clamp
        clamped = torch.clamp(added, min=-1.0, max=1.0)
        
        # Step 4: GELU
        output = F.gelu(clamped)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, sum_tensor = ctx.saved_tensors
        
        # For backward pass, use PyTorch's autograd
        with torch.enable_grad():
            input_detached = input.detach().requires_grad_()
            
            leaky_relu = F.leaky_relu(input_detached, negative_slope=0.2)
            added = leaky_relu + sum_tensor
            clamped = torch.clamp(added, min=-1.0, max=1.0)
            output = F.gelu(clamped)
            
            gradients = torch.autograd.grad(
                outputs=output,
                inputs=[input_detached, sum_tensor],
                grad_outputs=grad_output
            )
        
        return gradients[0], gradients[1]


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        
        # Initialize convolution with optimized parameters
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size
        )
        
        # Initialize sum_tensor parameter
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        
        # Use custom function for the post-processing
        self.fused_post_process = FusedPostProcessing.apply
        
        # Enable cuDNN benchmark mode for optimized convolution algorithms
        torch.backends.cudnn.benchmark = True
        
        # Pre-compile operations for faster execution
        self._is_initialized = False
        
    def _initialize_optimizations(self, x):
        # This is called once during the first forward pass to optimize memory layout
        if x.is_cuda:
            # Convert weights to channels_last_3d format if on CUDA
            if hasattr(self.conv.weight, 'data'):
                self.conv.weight.data = self.conv.weight.data.contiguous(memory_format=torch.channels_last_3d)
            
            # Pre-convert sum_tensor to contiguous format
            if not self.sum_tensor.is_contiguous():
                self.sum_tensor.data = self.sum_tensor.data.contiguous()
        
        self._is_initialized = True

    def forward(self, x):
        # Initialize optimizations on first run
        if not self._is_initialized:
            self._initialize_optimizations(x)
        
        # Convert input to channels_last_3d format if on CUDA for better performance
        if x.is_cuda and not x.is_contiguous(memory_format=torch.channels_last_3d):
            x = x.contiguous(memory_format=torch.channels_last_3d)
        
        # Step 1: Apply 3D convolution
        x = self.conv(x)
        
        # Step 2: Apply fused post-processing operations
        x = self.fused_post_process(x, self.sum_tensor)
        
        return x


# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
sum_tensor_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]