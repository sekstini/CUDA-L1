import torch
import torch.nn as nn
import math

class FusedMinSumGeluBias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        # Save input for backward pass
        ctx.save_for_backward(x, bias)
        
        # Ensure tensors are contiguous for better memory access
        x_cont = x if x.is_contiguous() else x.contiguous()
        
        # Step 1: Find min along channel dimension using amin (faster than min)
        min_vals = torch.amin(x_cont, dim=1, keepdim=True)
        
        # Step 2: Sum along height dimension
        sum_vals = torch.sum(min_vals, dim=2, keepdim=True)
        
        # Step 3: Apply GELU activation using optimized implementation
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        x_cubed = sum_vals * sum_vals * sum_vals
        inner = sqrt_2_over_pi * (sum_vals + 0.044715 * x_cubed)
        tanh_val = torch.tanh(inner)
        gelu_output = 0.5 * sum_vals * (1.0 + tanh_val)
        
        # Step 4: Add bias
        result = gelu_output + bias
        
        # Cache intermediate values for backward pass
        ctx.min_vals = min_vals
        ctx.sum_vals = sum_vals
        ctx.tanh_val = tanh_val
        ctx.inner = inner
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        x, bias = ctx.saved_tensors
        min_vals = ctx.min_vals
        sum_vals = ctx.sum_vals
        tanh_val = ctx.tanh_val
        inner = ctx.inner
        
        # Ensure grad_output is contiguous
        grad_output_cont = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
        
        # Gradient for bias - sum across batch and spatial dimensions
        grad_bias = grad_output_cont.sum(dim=(0, 2, 3), keepdim=True)
        
        # Compute GELU gradient
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        # d(tanh)/d(inner)
        dtanh = 1.0 - tanh_val * tanh_val
        # d(inner)/d(x)
        dinner_dx = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * sum_vals * sum_vals)
        # d(GELU)/d(x)
        dgelu_dx = 0.5 * (1.0 + tanh_val) + 0.5 * sum_vals * dtanh * dinner_dx
        
        # Apply chain rule with incoming gradient
        dsum_dx = dgelu_dx * grad_output_cont
        
        # Expand gradient to match height dimension for the sum operation
        batch_size, _, height, width = x.shape
        height_grad = dsum_dx.expand(-1, -1, height, -1)
        
        # Find which elements were the minimum
        is_min = (x == min_vals.expand_as(x))
        
        # Count how many elements achieved the minimum
        min_count = is_min.sum(dim=1, keepdim=True).clamp(min=1.0)
        
        # Create a normalized mask to distribute gradients
        normalized_mask = is_min.float() / min_count
        
        # Apply the mask to distribute gradients through the min operation
        grad_input = normalized_mask * height_grad
        
        return grad_input, grad_bias


class ModelNew(nn.Module):
    """
    An optimized model that performs a convolution transpose, minimum operation,
    sum operation, GELU activation and addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = FusedMinSumGeluBias.apply
        
        # Enable cuDNN benchmarking for optimal algorithm selection
        torch.backends.cudnn.benchmark = True
    
    def forward(self, x):
        # Use mixed precision where beneficial
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # Step 1: Perform ConvTranspose2d
            x = self.conv_transpose(x)
            
            # Steps 2-5: Use our optimized fused operations
            return self.fused_op(x, self.bias)


# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]