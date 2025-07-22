import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        scale_factor (float): Scaling factor to apply
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        # Initialize convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        
        # Enable cuDNN benchmarking for optimal convolution algorithm selection
        torch.backends.cudnn.benchmark = True
        
        # Fuse scaling into convolution weights and bias to eliminate separate scaling step
        with torch.no_grad():
            self.conv.weight.data.mul_(self.scale_factor)
            if self.conv.bias is not None:
                self.conv.bias.data.mul_(self.scale_factor)
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, height', width')
        """
        # Ensure input is contiguous for optimal convolution performance
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Step 1: Perform convolution (scaling is pre-fused into weights)
        x = self.conv(x)
        
        # Step 2: Use optimized amin function for channel-wise minimum reduction
        return torch.amin(x, dim=1, keepdim=True)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, scale_factor]