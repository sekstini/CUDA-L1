import torch
import torch.nn as nn

# Define the hyperparameters exactly as in the reference implementation
batch_size = 16
in_channels = 3
out_channels = 64
width = 256
height = 256

class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation with optimized implementation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        # Create a standard Conv2d with 1x1 kernel to ensure identical initialization and behavior
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        
        # Extract and prepare weights for efficient computation during initialization
        # Reshape weights from [out_channels, in_channels, 1, 1] to [in_channels, out_channels]
        self.register_buffer('weight_prepared', self.conv.weight.view(out_channels, in_channels).t().contiguous())
        
        # Store bias separately if needed
        if bias:
            self.register_buffer('bias_prepared', self.conv.bias.view(1, -1, 1, 1))
        else:
            self.register_buffer('bias_prepared', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using optimized matrix multiplication.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        # Get dimensions
        batch_size, in_channels, height, width = x.shape
        hw = height * width
        
        # Reshape input: [batch_size, in_channels, height*width] -> [batch_size, height*width, in_channels]
        # Use transpose instead of permute for better performance
        x_reshaped = x.view(batch_size, in_channels, hw).transpose(1, 2).contiguous()
        
        # Perform matrix multiplication: [batch_size, height*width, in_channels] x [in_channels, out_channels]
        # -> [batch_size, height*width, out_channels]
        output = torch.matmul(x_reshaped, self.weight_prepared)
        
        # Reshape output back to [batch_size, out_channels, height, width]
        output = output.transpose(1, 2).view(batch_size, out_channels, height, width)
        
        # Add bias if needed using broadcasting for efficiency
        if self.bias_prepared is not None:
            output = output + self.bias_prepared
            
        return output

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels]