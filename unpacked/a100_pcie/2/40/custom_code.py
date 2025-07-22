import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    """
    Your optimized implementation here that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features  
        scaling_factor (float): Scaling factor to apply
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        
        # Pre-compute the combined scaling factor
        combined_factor = 1.0 + scaling_factor
        
        # Create temporary weight and bias for initialization
        weight = torch.empty(out_features, in_features)
        bias = torch.empty(out_features)
        
        # Initialize parameters the same way nn.Linear would
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        
        # Pre-compute and store the scaled bias
        self.register_buffer('bias', bias * combined_factor)
        
        # Pre-transpose and make contiguous for optimal memory access patterns
        # Store the transposed weight matrix directly for efficient addmm
        self.register_buffer('weight_t', (weight * combined_factor).t().contiguous())
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Use torch.addmm for maximum efficiency - combines matrix multiplication and bias addition
        # This performs: bias + x @ weight_t in a single optimized CUDA kernel
        return torch.addmm(self.bias, x, self.weight_t)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 64
out_features = 128
scaling_factor = 0.5

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation  
    return [in_features, out_features, scaling_factor]