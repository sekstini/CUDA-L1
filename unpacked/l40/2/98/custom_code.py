import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    An optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        pool_kernel_size (int): Kernel size for average pooling
        scale_factor (float): Scaling factor to apply
    """
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        
        # Calculate the pooled output size
        self.pooled_size = out_features // pool_kernel_size
        
        # Create a standard linear layer for proper initialization
        temp_linear = nn.Linear(in_features, out_features)
        
        # Pre-compute pooled weights by reshaping and averaging
        # Shape: [out_features, in_features] -> [pooled_size, pool_kernel_size, in_features]
        w_reshaped = temp_linear.weight.view(self.pooled_size, pool_kernel_size, in_features)
        pooled_weight = w_reshaped.mean(dim=1)
        
        # Pre-compute pooled bias if present
        if temp_linear.bias is not None:
            b_reshaped = temp_linear.bias.view(self.pooled_size, pool_kernel_size)
            pooled_bias = b_reshaped.mean(dim=1)
        else:
            pooled_bias = None
        
        # Register the pooled parameters
        self.weight = nn.Parameter(pooled_weight)
        self.bias = nn.Parameter(pooled_bias)
        
        # Pre-compute the scaled factor for efficiency
        self.register_buffer('scaled_factor', torch.tensor(self.scale_factor, dtype=torch.float))
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Combined linear transformation and pooling using pre-computed weights
        # This single operation replaces both the linear and pooling steps
        pooled = F.linear(x, self.weight, self.bias)
        
        # GELU activation (using the built-in function for optimal CUDA implementation)
        activated = F.gelu(pooled)
        
        # Scale the result (in-place to reduce memory allocation)
        activated.mul_(self.scaled_factor)
        
        # Max reduction along dimension 1
        result = torch.max(activated, dim=1).values
        
        return result

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 512
out_features = 256
pool_kernel_size = 4
scale_factor = 2.0

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_features, out_features, pool_kernel_size, scale_factor]