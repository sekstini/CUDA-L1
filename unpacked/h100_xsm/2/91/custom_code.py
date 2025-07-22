import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized implementation of a model that performs transposed convolution,
    applies softmax, adds a bias term, scales the result, and applies sigmoid.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        stride (int): Stride of the convolution
        padding (int): Padding added to input
        output_padding (int): Additional padding for output
        bias_shape (tuple): Shape of the bias tensor
        scaling_factor (float): Scaling factor to apply
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        # Create a standard ConvTranspose2d layer
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        
        # Create bias parameter with the specified shape
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store scaling factor
        self.scaling_factor = scaling_factor
        
        # Initialize compilation flags
        self.use_compiled_full = False
        self.use_compiled_post = False
        
        # Define the full forward function for compilation
        def _full_forward(module, x):
            # Step 1: Perform transposed convolution
            x = module.conv_transpose(x)
            
            # Step 2: Softmax along channel dimension (with numerical stability)
            max_vals, _ = torch.max(x, dim=1, keepdim=True)
            x_sub = x - max_vals
            exp_x = torch.exp(x_sub)
            sum_exp = torch.sum(exp_x, dim=1, keepdim=True)
            x = exp_x / sum_exp
            
            # Step 3-5: Add bias, scale, and apply sigmoid
            return torch.sigmoid((x + module.bias) * module.scaling_factor)
        
        # Define post-convolution operations for separate compilation
        def _post_conv_ops(x, bias, scaling_factor):
            # Step 1: Softmax along channel dimension (with numerical stability)
            max_vals, _ = torch.max(x, dim=1, keepdim=True)
            x_sub = x - max_vals
            exp_x = torch.exp(x_sub)
            sum_exp = torch.sum(exp_x, dim=1, keepdim=True)
            x = exp_x / sum_exp
            
            # Step 2-4: Add bias, scale, and apply sigmoid
            return torch.sigmoid((x + bias) * scaling_factor)
        
        # Try to compile functions with different optimization levels
        try:
            # Try to compile the full forward function
            self._compiled_full_forward = torch.compile(
                _full_forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False
            )
            self.use_compiled_full = True
        except Exception:
            pass
            
        try:
            # Try to compile the post-convolution operations separately
            self._compiled_post_conv_ops = torch.compile(
                _post_conv_ops,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False
            )
            self.use_compiled_post = True
        except Exception:
            pass
            
        # Store non-compiled functions for fallback
        self._full_forward = _full_forward
        self._post_conv_ops = _post_conv_ops
    
    def _fallback_forward(self, x):
        """Fallback implementation using standard PyTorch operations"""
        # Step 1: Perform transposed convolution
        x = self.conv_transpose(x)
        
        # Step 2: Softmax along channel dimension (with numerical stability)
        max_vals, _ = torch.max(x, dim=1, keepdim=True)
        x_sub = x - max_vals
        exp_x = torch.exp(x_sub)
        sum_exp = torch.sum(exp_x, dim=1, keepdim=True)
        x = exp_x / sum_exp
        
        # Step 3: Add bias
        x = x + self.bias
        
        # Step 4: Scale
        x = x * self.scaling_factor
        
        # Step 5: Sigmoid
        return torch.sigmoid(x)
    
    def forward(self, x):
        with torch.no_grad():  # Disable gradient computation for inference
            # Try using the fully compiled forward function
            if self.use_compiled_full:
                try:
                    return self._compiled_full_forward(self, x)
                except Exception:
                    pass
            
            # Try using the compiled post-conv operations with standard convolution
            if self.use_compiled_post:
                try:
                    conv_output = self.conv_transpose(x)
                    return self._compiled_post_conv_ops(conv_output, self.bias, self.scaling_factor)
                except Exception:
                    pass
            
            # Fallback to non-compiled but optimized implementation
            try:
                conv_output = self.conv_transpose(x)
                return self._post_conv_ops(conv_output, self.bias, self.scaling_factor)
            except Exception:
                pass
            
            # Final fallback to basic implementation
            return self._fallback_forward(x)

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
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]