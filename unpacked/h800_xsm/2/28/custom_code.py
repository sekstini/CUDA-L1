import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        eps (float): Small constant added to the denominator for numerical stability
        momentum (float): The value used for the running_mean and running_var computation
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Create optimized linear transformation parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters exactly like nn.Linear for identical behavior
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        bound = 1 / (in_features**0.5)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Store eps for layer norm
        self.eps = eps
        
        # Pre-allocate normalized_shape for layer norm
        self.normalized_shape = (out_features,)
    
    def forward(self, x, y):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Input tensor of shape (batch_size, out_features).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Step 1: Optimized linear transformation using F.linear
        x = F.linear(x, self.weight, self.bias)
        
        # Step 2: Use F.layer_norm directly for maximum performance
        # This is mathematically equivalent to the instance norm operation
        x = F.layer_norm(x, self.normalized_shape, eps=self.eps)
        
        # Step 3: Optimized in-place operations for residual and multiplication
        # Use add_ and mul_ for maximum memory efficiency
        x = x.add_(y).mul_(y)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 64
out_features = 128

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features), torch.randn(batch_size, out_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_features, out_features]