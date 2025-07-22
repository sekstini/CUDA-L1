import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D transposed convolution, scales the output, applies batch normalization, 
    and then performs global average pooling.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolving kernel
        scale_factor (float): Scaling factor to apply to the output of convolution
        eps (float, optional): Value added to the denominator for numerical stability in batch norm
        momentum (float, optional): Value used for the running_mean and running_var computation
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        
        # Create the transposed convolution layer
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        
        # Create batch normalization layer
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        
        # Store the scale factor
        self.scale_factor = scale_factor
        
        # Fuse the scaling factor into the convolution weights and bias
        with torch.no_grad():
            # Scale the weights
            self.conv_transpose.weight.data *= scale_factor
            
            # Scale the bias if it exists
            if self.conv_transpose.bias is not None:
                self.conv_transpose.bias.data *= scale_factor
                
        # Convert weights to channels_last_3d format for better performance
        self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(
            memory_format=torch.channels_last_3d).contiguous()
        
        # Cache convolution parameters for efficient access
        self.weight = self.conv_transpose.weight
        self.bias = self.conv_transpose.bias
        self.stride = self.conv_transpose.stride
        self.padding = self.conv_transpose.padding
        self.output_padding = self.conv_transpose.output_padding
        self.groups = self.conv_transpose.groups
        self.dilation = self.conv_transpose.dilation
        
        # Create a CUDA stream for optimized execution
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
    
    def forward(self, x):
        # Use CUDA stream if available
        if self.stream is not None and x.is_cuda:
            with torch.cuda.stream(self.stream):
                return self._optimized_forward(x)
        else:
            return self._optimized_forward(x)
    
    def _optimized_forward(self, x):
        # Convert to channels_last_3d memory format for better performance
        x = x.to(memory_format=torch.channels_last_3d).contiguous()
        
        # Apply transposed convolution (with pre-fused scaling)
        x = F.conv_transpose3d(
            x, 
            self.weight, 
            self.bias,
            stride=self.stride, 
            padding=self.padding, 
            output_padding=self.output_padding,
            groups=self.groups, 
            dilation=self.dilation
        )
        
        # Apply batch normalization (keeping channels_last_3d format)
        x = self.batch_norm(x)
        
        # Apply global average pooling directly with mean operation
        # This is more efficient than AdaptiveAvgPool3d for pooling to (1,1,1)
        x = x.mean(dim=[2, 3, 4], keepdim=True)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 64
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation  
    return [in_channels, out_channels, kernel_size, scale_factor]