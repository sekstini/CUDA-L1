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
        # Initialize parameters similar to nn.Linear for exact compatibility
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters using the same method as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Store scaling factor as a buffer for efficient access
        self.register_buffer('scaling_factor', torch.tensor(scaling_factor, dtype=torch.float))
        
        # Pre-transpose weight for faster matmul in forward pass
        self.register_buffer('weight_t', self.weight.t().contiguous(), persistent=False)
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Use addmm for efficient fused linear transformation
        # out = bias + (x @ weight_t)
        out = torch.addmm(self.bias, x, self.weight_t)
        
        # Compute Swish activation (x * sigmoid(x)) and scaling in one operation chain
        # Using in-place operations to minimize memory allocations
        out.mul_(torch.sigmoid(out)).mul_(self.scaling_factor)
        
        return out

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 2.0

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation  
    return [in_features, out_features, scaling_factor]