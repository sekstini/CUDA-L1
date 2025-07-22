import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OptimizedConvTranspose3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, output_padding):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        
        # Use PyTorch's optimized implementation for forward pass
        output = F.conv_transpose3d(input, weight, bias, stride, padding, output_padding)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        output_padding = ctx.output_padding
        
        # Calculate input gradient using PyTorch's optimized function
        grad_input = F.conv3d(grad_output, weight, None, stride, padding)
        
        # Calculate weight gradient efficiently
        batch_size = input.size(0)
        in_channels = input.size(1)
        out_channels = grad_output.size(1)
        
        # Initialize grad_weight with zeros
        grad_weight = torch.zeros_like(weight)
        
        # Vectorized implementation for weight gradient
        for b in range(batch_size):
            for c_in in range(in_channels):
                for c_out in range(out_channels):
                    # Extract slices
                    input_slice = input[b, c_in].unsqueeze(0).unsqueeze(0)
                    grad_output_slice = grad_output[b, c_out].unsqueeze(0).unsqueeze(0)
                    
                    # Calculate gradient contribution
                    grad_weight[c_in, c_out] += F.conv3d(input_slice, grad_output_slice, None, stride, padding)
        
        # Calculate bias gradient if bias is not None
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=(0, 2, 3, 4))
        
        return grad_input, grad_weight, grad_bias, None, None, None

class FusedPostProcessing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, pool_kernel_size, clamp_min, clamp_max):
        # Save parameters for backward pass
        ctx.pool_kernel_size = pool_kernel_size
        ctx.clamp_min = clamp_min
        ctx.clamp_max = clamp_max
        
        # 1. Average pooling
        pooled = F.avg_pool3d(input, pool_kernel_size)
        
        # 2. Clamping
        clamped = torch.clamp(pooled, clamp_min, clamp_max)
        
        # 3. Softmax
        softmaxed = F.softmax(clamped, dim=1)
        
        # 4. Multiplication
        output = softmaxed * 2
        
        # Save intermediate results for backward pass
        ctx.save_for_backward(input, pooled, clamped, softmaxed)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, pooled, clamped, softmaxed = ctx.saved_tensors
        pool_kernel_size = ctx.pool_kernel_size
        clamp_min = ctx.clamp_min
        clamp_max = ctx.clamp_max
        
        # Backward for multiplication by 2
        grad_softmax = grad_output * 2
        
        # Backward for softmax
        grad_clamped = torch._softmax_backward_data(grad_softmax, softmaxed, 1, softmaxed.dtype)
        
        # Backward for clamp
        grad_pooled = grad_clamped * ((clamped > clamp_min) & (clamped < clamp_max)).float()
        
        # Backward for avg_pool3d
        # Use efficient upsampling for the gradient
        batch_size, channels, pooled_d, pooled_h, pooled_w = pooled.shape
        input_d, input_h, input_w = input.shape[2], input.shape[3], input.shape[4]
        
        # Create a tensor of the right shape filled with the gradient
        grad_input = torch.zeros_like(input)
        
        # Distribute the gradient evenly across the pooling window
        scale_factor = 1.0 / (pool_kernel_size ** 3)
        
        # Vectorized implementation for better performance
        for d in range(pooled_d):
            d_start = d * pool_kernel_size
            d_end = min(d_start + pool_kernel_size, input_d)
            d_size = d_end - d_start
            
            for h in range(pooled_h):
                h_start = h * pool_kernel_size
                h_end = min(h_start + pool_kernel_size, input_h)
                h_size = h_end - h_start
                
                for w in range(pooled_w):
                    w_start = w * pool_kernel_size
                    w_end = min(w_start + pool_kernel_size, input_w)
                    w_size = w_end - w_start
                    
                    # Calculate actual window size (handling edge cases)
                    window_size = d_size * h_size * w_size
                    actual_scale = scale_factor * (pool_kernel_size**3 / window_size) if window_size > 0 else 0
                    
                    # Distribute gradient to all elements in the pooling window using broadcasting
                    grad_input[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += (
                        grad_pooled[:, :, d, h, w].reshape(batch_size, channels, 1, 1, 1) * actual_scale
                    )
        
        return grad_input, None, None, None

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D transposed convolution, average pooling, clamping, softmax, and multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        
        # Custom operations
        self.optimized_conv_transpose = OptimizedConvTranspose3d.apply
        self.fused_post_processing = FusedPostProcessing.apply
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        # Use optimized operations if on CUDA
        if x.is_cuda:
            with torch.cuda.stream(torch.cuda.Stream()):
                # Step 1: Optimized ConvTranspose3d
                x = self.optimized_conv_transpose(
                    x, self.weight, self.bias, 
                    self.stride, self.padding, self.output_padding
                )
                
                # Step 2: Fused post-processing operations
                x = self.fused_post_processing(
                    x, self.pool_kernel_size, self.clamp_min, self.clamp_max
                )
                
                return x
        else:
            # Fallback to sequential processing for CPU
            x = F.conv_transpose3d(
                x, self.weight, self.bias, 
                self.stride, self.padding, self.output_padding
            )
            x = F.avg_pool3d(x, self.pool_kernel_size)
            x = torch.clamp(x, self.clamp_min, self.clamp_max)
            x = F.softmax(x, dim=1)
            x = x * 2
            return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]