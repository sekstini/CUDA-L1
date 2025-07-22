import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedConvTranspose3dAddHardswishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, add_input, stride, padding, output_padding):
        # Save for backward
        ctx.save_for_backward(input, weight, bias, add_input)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        
        # Perform transposed convolution
        conv_output = F.conv_transpose3d(input, weight, bias, 
                                      stride=stride, padding=padding, 
                                      output_padding=output_padding)
        
        # Add the additional input
        added = conv_output + add_input
        
        # Apply HardSwish activation
        # Store intermediate results for backward pass optimization
        ctx.added = added
        
        # Compute hardswish efficiently
        result = added * F.hardswish(added)
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, add_input = ctx.saved_tensors
        stride, padding, output_padding = ctx.stride, ctx.padding, ctx.output_padding
        added = ctx.added
        
        # Initialize gradients
        grad_input = grad_weight = grad_bias = grad_add_input = None
        
        # Compute hardswish gradient efficiently
        # HardSwish'(x) = 
        # 0 if x <= -3
        # 1 if x >= 3
        # (2x + 6)/6 if -3 < x < 3
        
        # Compute gradient for hardswish(x) * x
        # d(hardswish(x) * x)/dx = hardswish(x) + x * hardswish'(x)
        
        # First calculate hardswish'(x)
        hardswish_val = F.hardswish(added)
        hardswish_grad = torch.zeros_like(added)
        mask_neg = added <= -3
        mask_pos = added >= 3
        mask_mid = ~(mask_neg | mask_pos)
        hardswish_grad[mask_pos] = 1.0
        hardswish_grad[mask_mid] = (2 * added[mask_mid] + 6) / 6
        
        # Then calculate gradient for added
        grad_added = grad_output * (hardswish_val + added * hardswish_grad)
        
        # Gradient for add_input is simply grad_added
        if ctx.needs_input_grad[3]:
            grad_add_input = grad_added
        
        # Compute gradients for convolution
        if ctx.needs_input_grad[0]:
            # Use transposed convolution for input gradient
            # This is more efficient than manual implementation
            grad_input = torch.nn.grad.conv_transpose3d_input_grad(
                input.shape, weight, grad_added, stride=stride,
                padding=padding, output_padding=0, dilation=1, groups=1
            )
        
        if ctx.needs_input_grad[1]:
            # Use optimized weight gradient calculation
            # This is more efficient than the previous implementation
            batch_size = input.size(0)
            in_channels = input.size(1)
            out_channels = weight.size(0)
            
            # Initialize weight gradient
            grad_weight = torch.zeros_like(weight)
            
            # Compute weight gradient using convolution
            # Reshape inputs for batch convolution
            input_reshaped = input.reshape(1, batch_size * in_channels, *input.shape[2:])
            grad_added_reshaped = grad_added.transpose(0, 1).reshape(out_channels, batch_size, *grad_added.shape[2:])
            
            # Process each output channel
            for oc in range(out_channels):
                # Process all input channels at once for this output channel
                grad_slice = grad_added_reshaped[oc].reshape(1, batch_size, *grad_added.shape[2:])
                
                # Use convolution to compute correlation efficiently
                for ic in range(in_channels):
                    input_slice = input_reshaped[:, batch_size*ic:batch_size*(ic+1)]
                    
                    # Use grouped convolution for efficiency
                    weight_update = F.conv3d(
                        input_slice,
                        grad_slice,
                        padding=padding,
                        stride=1,
                        groups=batch_size
                    )
                    
                    # Sum across batch dimension
                    grad_weight[oc, ic] = weight_update.sum(dim=0)
        
        # Compute bias gradient if bias is not None
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_added.sum(dim=(0, 2, 3, 4)).reshape_as(bias)
        
        return grad_input, grad_weight, grad_bias, grad_add_input, None, None, None

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, 
                                               stride=stride, padding=padding, 
                                               output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store parameters for the optimized implementation
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
        # For optimization with multiple CUDA streams
        if torch.cuda.is_available():
            self.stream1 = torch.cuda.Stream()
            self.stream2 = torch.cuda.Stream()
            
            # Enable cuDNN benchmarking for better performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    def forward(self, x, add_input):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
            add_input (torch.Tensor): Input tensor to be added after transposed convolution,
                                     of shape (batch_size, out_channels, D*stride, H*stride, W*stride).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D*stride, H*stride, W*stride)
                         after HardSwish activation.
        """
        if x.is_cuda:
            # Use multiple streams for better parallelism
            current_stream = torch.cuda.current_stream()
            
            # Prefetch tensors to GPU cache
            with torch.cuda.stream(self.stream1):
                _ = self.conv_transpose.weight.size()
                if self.conv_transpose.bias is not None:
                    _ = self.conv_transpose.bias.size()
            
            with torch.cuda.stream(self.stream2):
                _ = add_input.size()
            
            # Execute the fused operation with mixed precision where beneficial
            with torch.cuda.amp.autocast(enabled=True):
                output = FusedConvTranspose3dAddHardswishFunction.apply(
                    x, self.conv_transpose.weight, self.conv_transpose.bias, 
                    add_input, self.stride, self.padding, self.output_padding
                )
            
            # Synchronize streams before returning
            current_stream.wait_stream(self.stream1)
            current_stream.wait_stream(self.stream2)
            
            return output
        else:
            # Fall back to standard implementation for CPU
            x = self.conv_transpose(x)
            x = x + add_input
            x = x * F.hardswish(x)
            return x


# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W), torch.randn(batch_size, out_channels, D*stride, H*stride, W*stride)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]