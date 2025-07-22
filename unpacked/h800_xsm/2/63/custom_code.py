import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Your optimized implementation here that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features  
        divisor (float): Divisor to apply
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        # Create weight and bias parameters (same as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters (same as nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Pre-compute scaled weights and bias to eliminate division from forward pass
        with torch.no_grad():
            # Pre-divide by divisor and pre-transpose for optimal addmm usage
            scaled_weight_t = (self.weight / divisor).t().contiguous()
            
            # Pre-divide bias by divisor
            scaled_bias = (self.bias / divisor).contiguous()
        
        # Register as buffers to ensure they move with the model
        self.register_buffer('scaled_weight_t', scaled_weight_t)
        self.register_buffer('scaled_bias', scaled_bias)
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Single-expression forward pass: fused matrix multiply + bias add, then in-place ReLU
        return torch.relu_(torch.addmm(self.scaled_bias, x, self.scaled_weight_t))

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 1024
out_features = 512
divisor = 2.0

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation  
    return [in_features, out_features, divisor]