import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelNew(nn.Module):
    """
    An optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        input_size (int): Number of input features
        output_size (int): Number of output features  
        divisor (float): Divisor to apply
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        # Create weight and bias parameters directly
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        
        # Initialize parameters same as nn.Linear would
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Pre-divide weights and bias by the divisor
        with torch.no_grad():
            self.weight.div_(divisor)
            self.bias.div_(divisor)
        
        # Pre-transpose weight for more efficient matrix multiplication
        self.register_buffer('weight_t', self.weight.t().contiguous())
        
        # Register a hook to update weight_t when weight changes
        self.weight.register_hook(self._weight_hook)
    
    def _weight_hook(self, grad):
        # Update the transposed weight when the weight is updated
        with torch.no_grad():
            self.weight_t.copy_(self.weight.t().contiguous())
        return grad
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        # Use torch.addmm for fused matrix multiplication and bias addition
        # followed directly by GELU activation with no intermediate allocations
        return F.gelu(torch.addmm(self.bias, x, self.weight_t))

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_size = 512
output_size = 1024
divisor = 10.0

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation  
    return [input_size, output_size, divisor]