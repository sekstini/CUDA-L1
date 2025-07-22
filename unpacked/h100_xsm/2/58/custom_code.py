import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function

class ConvTranspose3dFusedFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        # Save tensors needed for backward pass
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        
        # Ensure input is contiguous for better memory access patterns
        input = input.contiguous()
        weight = weight.contiguous()
        
        # Use PyTorch's optimized conv_transpose3d
        conv_output = F.conv_transpose3d(input, weight, None, stride, padding, 0, 1)
        
        # Fused logsumexp computation with numerical stability
        max_vals, _ = torch.max(conv_output, dim=1, keepdim=True)
        exp_shifted = torch.exp(conv_output - max_vals)
        sum_exp = torch.sum(exp_shifted, dim=1, keepdim=True)
        logsumexp = max_vals + torch.log(sum_exp)
        
        # Free memory early
        del conv_output, exp_shifted
        
        # Fused HardSwish: x * sigmoid(x + 3) / 6
        # Using multiplication instead of division for better performance
        sigmoid_val = torch.sigmoid(logsumexp + 3.0)
        hardswish = logsumexp * sigmoid_val * (1.0/6.0)
        
        # Free memory early
        del logsumexp, sigmoid_val
        
        # Fused bias subtraction, clamp, and max operations
        result = hardswish - bias
        del hardswish
        
        clamped = torch.clamp(result, min=-1.0, max=1.0)
        del result
        
        output, _ = torch.max(clamped, dim=1, keepdim=True)
        del clamped
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        
        # Initialize gradients
        grad_input = grad_weight = grad_bias = None
        
        # Only compute gradients for tensors that require them
        needs_input_grad = ctx.needs_input_grad[0]
        needs_weight_grad = ctx.needs_input_grad[1]
        needs_bias_grad = ctx.needs_input_grad[2]
        
        # Skip computation if no gradients are needed
        if not (needs_input_grad or needs_weight_grad or needs_bias_grad):
            return None, None, None, None, None
        
        # Create tensors that require gradients with minimal overhead
        input_new = input.detach().requires_grad_(needs_input_grad)
        weight_new = weight.detach().requires_grad_(needs_weight_grad)
        bias_new = bias.detach().requires_grad_(needs_bias_grad)
        
        with torch.enable_grad():
            # Forward pass with minimal temporary tensor creation
            conv_out = F.conv_transpose3d(input_new, weight_new, None, stride, padding, 0, 1)
            
            # Optimize memory usage by reusing tensors where possible
            max_vals, indices = torch.max(conv_out, dim=1, keepdim=True)
            
            # Use in-place operations where possible to reduce memory allocations
            exp_shifted = torch.exp(conv_out - max_vals)
            sum_exp = torch.sum(exp_shifted, dim=1, keepdim=True)
            logsumexp = max_vals + torch.log(sum_exp)
            
            # Free memory early
            del max_vals, sum_exp, exp_shifted
            
            sigmoid_input = logsumexp + 3.0
            sigmoid_val = torch.sigmoid(sigmoid_input)
            hardswish = logsumexp * sigmoid_val * (1.0/6.0)
            
            # Free memory early
            del logsumexp, sigmoid_input, sigmoid_val
            
            result = hardswish - bias_new
            del hardswish
            
            clamped = torch.clamp(result, min=-1.0, max=1.0)
            del result
            
            output, _ = torch.max(clamped, dim=1, keepdim=True)
            del clamped
            
            # Efficient backward pass
            output.backward(grad_output)
            
            # Extract gradients only if needed
            if needs_input_grad:
                grad_input = input_new.grad.detach()
            if needs_weight_grad:
                grad_weight = weight_new.grad.detach()
            if needs_bias_grad:
                grad_bias = bias_new.grad.detach()
        
        return grad_input, grad_weight, grad_bias, None, None

class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, LogSumExp, HardSwish, subtraction, clamp, and maximum operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        
        # Initialize weights similar to nn.ConvTranspose3d with optimized initialization
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, *self.kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Initialize bias with proper shape
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Create CUDA stream for optimized execution
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
        
    def forward(self, x):
        # Ensure input is contiguous for optimal memory access
        x = x.contiguous()
        
        # Use CUDA stream for asynchronous execution if available
        if self.stream is not None and x.is_cuda:
            with torch.cuda.stream(self.stream):
                return ConvTranspose3dFusedFunction.apply(
                    x, self.weight, self.bias, self.stride, self.padding
                )
        else:
            return ConvTranspose3dFusedFunction.apply(
                x, self.weight, self.bias, self.stride, self.padding
            )

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]