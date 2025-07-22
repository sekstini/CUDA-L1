import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedFusedConvTransposeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, output_padding, scaling_factor):
        # Save parameters for backward pass
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.scaling_factor = scaling_factor
        
        # Step 1: Apply transposed convolution
        conv_output = F.conv_transpose2d(x, weight, None, stride, padding, output_padding)
        
        # Step 2: Optimized softmax computation with numerical stability
        # Use amax for better performance (proven in No2)
        max_vals = torch.amax(conv_output, dim=1, keepdim=True)
        
        # Compute exp(x - max) in-place for memory efficiency
        exp_vals = torch.exp(conv_output - max_vals)
        
        # Compute sum along channel dimension
        sum_exp = torch.sum(exp_vals, dim=1, keepdim=True)
        
        # Compute softmax: exp(x - max) / sum(exp(x - max))
        # Reuse exp_vals tensor to save memory
        softmax_output = exp_vals.div_(sum_exp)
        
        # Step 3-5: Fused bias addition, scaling, and sigmoid
        # Combine operations to minimize memory accesses
        # Use broadcasting efficiently for bias addition
        biased_scaled = (softmax_output + bias) * scaling_factor
        result = torch.sigmoid(biased_scaled)
        
        # Save minimal tensors for backward pass
        ctx.save_for_backward(x, weight, bias, softmax_output.detach())
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias, softmax_output = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        output_padding = ctx.output_padding
        scaling_factor = ctx.scaling_factor
        
        # Initialize gradients
        grad_x = grad_weight = grad_bias = None
        
        # Recompute intermediate values efficiently
        biased_scaled = (softmax_output + bias) * scaling_factor
        sigmoid_output = torch.sigmoid(biased_scaled)
        
        # Gradient through sigmoid: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        sigmoid_grad = sigmoid_output * (1.0 - sigmoid_output)
        grad_sigmoid = grad_output * sigmoid_grad
        
        # Gradient through scaling
        grad_scaled = grad_sigmoid * scaling_factor
        
        # Gradient for bias - optimized computation
        if ctx.needs_input_grad[2]:
            grad_bias = torch.sum(grad_scaled, dim=(0, 2, 3), keepdim=True)
        
        # For softmax and conv_transpose gradients, use optimized autograd
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            # Create computation graph with minimal overhead
            with torch.enable_grad():
                # Only create gradients for needed tensors
                x_grad = x.detach().requires_grad_(ctx.needs_input_grad[0]) if ctx.needs_input_grad[0] else x.detach()
                weight_grad = weight.detach().requires_grad_(ctx.needs_input_grad[1]) if ctx.needs_input_grad[1] else weight.detach()
                
                # Efficient forward pass for gradient computation
                conv_out = F.conv_transpose2d(x_grad, weight_grad, None, stride, padding, output_padding)
                softmax_out = F.softmax(conv_out, dim=1)
                
                # Backward pass with optimized gradient computation
                softmax_out.backward(grad_scaled, retain_graph=False)
                
                # Extract gradients
                grad_x = x_grad.grad if ctx.needs_input_grad[0] else None
                grad_weight = weight_grad.grad if ctx.needs_input_grad[1] else None
        
        return grad_x, grad_weight, grad_bias, None, None, None, None

class ModelNew(nn.Module):
    """
    Optimized implementation of the model that performs a transposed convolution,
    applies softmax, adds a bias term, scales the result, and applies sigmoid.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolving kernel
        stride (int): Stride of the convolution
        padding (int): Padding added to input
        output_padding (int): Additional size added to output
        bias_shape (tuple): Shape of the bias tensor
        scaling_factor (float): Scaling factor to apply
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        
        # Cache parameters for the fused function
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        # Optimization flag
        self.use_optimized = True
    
    def forward(self, x):
        if self.use_optimized:
            try:
                # Use our optimized fused operation
                return OptimizedFusedConvTransposeFunction.apply(
                    x, self.conv_transpose.weight, self.bias,
                    self.stride, self.padding, self.output_padding,
                    self.scaling_factor
                )
            except Exception:
                # Fallback to standard implementation if needed
                self.use_optimized = False
        
        # Standard implementation fallback
        x = self.conv_transpose(x)
        x = torch.softmax(x, dim=1)
        x = x + self.bias
        x = x * self.scaling_factor
        x = torch.sigmoid(x)
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]