import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication, applies sigmoid, and sums the result.
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden features
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        # Store weights and bias directly as parameters
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        
        # Initialize parameters (same as nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Pre-transpose weight for more efficient computation
        self.register_buffer('weight_t', self.weight.t().contiguous())
        
        # Flag to track if weight has been updated
        self._need_weight_t_update = False
        
        # Register a hook to update weight_t when weight changes
        def hook(grad):
            self._need_weight_t_update = True
            return None
        
        self.weight.register_hook(hook)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        # Update transposed weight if needed
        if self._need_weight_t_update:
            with torch.no_grad():
                self.weight_t.copy_(self.weight.t().contiguous())
            self._need_weight_t_update = False
        
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Fused matrix multiplication with bias addition
        # This is more efficient than separate matmul and bias addition
        out = torch.addmm(self.bias, x, self.weight_t)
        
        # Apply sigmoid activation
        out.sigmoid_()  # Inplace operation to reduce memory allocation
        
        # Sum along dimension 1 with keepdim=True
        return out.sum(dim=1, keepdim=True)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_size = 10
hidden_size = 20

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size]