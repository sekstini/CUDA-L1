import torch
import torch.nn as nn
import math

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
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize weights with the same shape as nn.Conv2d for compatibility
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
        # Pre-reshape weights for efficiency
        self.register_buffer('weight_reshaped', None, persistent=False)
        
    def reset_parameters(self):
        # Use the same initialization as nn.Conv2d for consistency
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using optimized implementation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        batch_size, in_channels, height, width = x.shape
        spatial_size = height * width
        
        # Ensure we have the reshaped weight tensor
        if self.weight_reshaped is None or self.weight_reshaped.shape[1] != self.out_channels:
            # Reshape weights: [out_channels, in_channels, 1, 1] -> [in_channels, out_channels]
            self.weight_reshaped = self.weight.view(self.out_channels, self.in_channels).t().contiguous()
        
        # Ensure input is contiguous for efficient reshaping
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Reshape input: [batch_size, in_channels, height, width] -> [batch_size, height*width, in_channels]
        x_reshaped = x.view(batch_size, in_channels, spatial_size).transpose(1, 2)
        
        # Perform batch matrix multiplication: 
        # [batch_size, height*width, in_channels] @ [in_channels, out_channels]
        # -> [batch_size, height*width, out_channels]
        # Create a batch-expanded weight tensor once to avoid repeated expansion
        weight_expanded = self.weight_reshaped.expand(batch_size, -1, -1)
        output = torch.bmm(x_reshaped, weight_expanded)
        
        # Reshape output: [batch_size, height*width, out_channels] -> [batch_size, out_channels, height, width]
        output = output.transpose(1, 2).view(batch_size, self.out_channels, height, width)
        
        # Add bias if needed
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
            
        return output

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
out_channels = 64
width = 256
height = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels]