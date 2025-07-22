import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.bias = bias
        
        # Create standard PyTorch ConvTranspose3d layer for weight and bias parameters
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, self.kernel_size,
            stride=self.stride, padding=self.padding,
            output_padding=self.output_padding, bias=bias
        )
        
        # Create CUDA stream for asynchronous execution
        self.stream = None
        if torch.cuda.is_available():
            try:
                self.stream = torch.cuda.Stream()
            except:
                self.stream = None
                
        # Cache weight and bias tensors
        self.weight_contiguous = None
        self.bias_tensor = None
        
        # Track weight updates
        self.weight_version = 0
        self.current_weight_version = -1
        
        # Register hook to track weight updates
        def hook(grad):
            self.weight_version += 1
            return grad
            
        if hasattr(self.conv_transpose, 'weight') and self.conv_transpose.weight.requires_grad:
            self.conv_transpose.weight.register_hook(hook)
            
        # Ensure weights are contiguous for better memory access
        if hasattr(self.conv_transpose, 'weight'):
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.contiguous()
            
        # Flag for optimization availability
        self.use_optimized = torch.cuda.is_available()

    def _optimized_softmax_sigmoid(self, x):
        """
        Highly optimized implementation of softmax followed by sigmoid
        """
        # Get shape information
        batch_size, channels, depth, height, width = x.shape
        
        # Ensure input is contiguous for better memory access
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Use view for better performance (no memory copy)
        x_flat = x.view(batch_size, channels, -1)
        
        # Find max for numerical stability (along channel dimension)
        x_max, _ = torch.max(x_flat, dim=1, keepdim=True)
        
        # Subtract max and compute exp (in-place operations)
        x_flat.sub_(x_max).exp_()
        
        # Compute sum and normalize (in-place division)
        x_sum = torch.sum(x_flat, dim=1, keepdim=True)
        x_flat.div_(x_sum)
        
        # Apply sigmoid (in-place)
        x_flat.sigmoid_()
        
        # Reshape back to original dimensions (view instead of reshape)
        return x_flat.view(batch_size, channels, depth, height, width)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        # Use optimized implementation if on CUDA
        if self.use_optimized and x.is_cuda:
            try:
                # Ensure input is contiguous for better memory access
                if not x.is_contiguous():
                    x = x.contiguous()
                
                # Check if weights have been updated since last cache
                if self.current_weight_version != self.weight_version or self.weight_contiguous is None:
                    weight = self.conv_transpose.weight
                    if not weight.is_contiguous():
                        self.weight_contiguous = weight.contiguous()
                    else:
                        self.weight_contiguous = weight
                    self.bias_tensor = self.conv_transpose.bias
                    self.current_weight_version = self.weight_version
                
                # Use stream if available
                if self.stream is not None:
                    with torch.cuda.stream(self.stream):
                        # Apply transposed convolution using F.conv_transpose3d directly
                        conv_output = F.conv_transpose3d(
                            x, self.weight_contiguous, self.bias_tensor,
                            self.stride, self.padding, self.output_padding, 
                            groups=1, dilation=1
                        )
                        
                        # Apply optimized softmax and sigmoid
                        result = self._optimized_softmax_sigmoid(conv_output)
                        
                        return result
                else:
                    # Direct execution without stream
                    conv_output = F.conv_transpose3d(
                        x, self.weight_contiguous, self.bias_tensor,
                        self.stride, self.padding, self.output_padding, 
                        groups=1, dilation=1
                    )
                    
                    result = self._optimized_softmax_sigmoid(conv_output)
                    
                    return result
                    
            except Exception as e:
                # Fallback to standard implementation if optimization fails
                self.use_optimized = False
        
        # Standard implementation as fallback
        x = self.conv_transpose(x)
        x = F.softmax(x, dim=1)
        x = torch.sigmoid(x)
        return x
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super(ModelNew, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        # Mark weights as updated when state_dict is loaded
        self.weight_version += 1

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]