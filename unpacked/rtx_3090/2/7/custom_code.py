import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedActivationsBias(torch.autograd.Function):
    """
    Custom autograd function that fuses ReLU, LeakyReLU, GELU, Sigmoid activations and bias addition.
    """
    @staticmethod
    def forward(ctx, input, bias):
        # Save input and bias for backward pass
        ctx.save_for_backward(input, bias)
        
        # Apply operations in sequence with minimal memory overhead
        # First apply ReLU (all values become non-negative)
        result = F.relu(input)
        
        # After ReLU, all values are non-negative, so LeakyReLU with slope=0.01 is redundant
        # for positive values. But we need to maintain the exact computation sequence
        # for the backward pass.
        
        # Apply GELU
        result = F.gelu(result)
        
        # Apply Sigmoid
        result = torch.sigmoid(result)
        
        # Add bias (broadcasting happens automatically)
        result = result + bias
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        grad_input = None
        grad_bias = None
        
        if ctx.needs_input_grad[0]:
            # Compute gradients through the chain of operations
            with torch.enable_grad():
                x = input.detach().requires_grad_()
                
                # Forward pass (must match the exact sequence in the reference implementation)
                relu_output = F.relu(x)
                leaky_relu_output = F.leaky_relu(relu_output, negative_slope=0.01)
                gelu_output = F.gelu(leaky_relu_output)
                sigmoid_output = torch.sigmoid(gelu_output)
                
                # Backward pass
                grad_input = torch.autograd.grad(sigmoid_output, x, grad_output)[0]
        
        if ctx.needs_input_grad[1]:
            # Gradient for bias is the sum of grad_output across all dimensions except channel
            grad_bias = grad_output.sum(dim=(0, 2, 3, 4), keepdim=True)
        
        return grad_input, grad_bias

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, 
    and bias in sequence with memory and computation optimizations.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolution kernel
        bias_shape (tuple): Shape of the bias tensor
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # Initialize convolution layer (no padding to match reference implementation)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        
        # Initialize bias parameter
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Initialize fused activation function
        self.fused_activations = FusedActivationsBias.apply
        
        # Pre-convert weights to optimal memory format if on CUDA
        if torch.cuda.is_available():
            self.conv.weight.data = self.conv.weight.data.to(memory_format=torch.channels_last_3d)
            
            # Enable cudnn benchmarking for faster convolution
            torch.backends.cudnn.benchmark = True
    
    def forward(self, x):
        # Convert to channels_last_3d memory format for optimal Conv3d performance if on CUDA
        if x.device.type == 'cuda':
            x = x.to(memory_format=torch.channels_last_3d)
            
            # Ensure weights are in optimal memory format
            if not self.conv.weight.is_contiguous(memory_format=torch.channels_last_3d):
                self.conv.weight.data = self.conv.weight.data.to(memory_format=torch.channels_last_3d)
        
        # Apply convolution
        x = self.conv(x)
        
        # Apply fused activation functions and bias addition
        x = self.fused_activations(x, self.bias)
        
        return x


# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, bias_shape]