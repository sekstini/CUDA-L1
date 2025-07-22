import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved performance through aggressive memory optimization
    and operation fusion.
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden features
        output_size (int): Number of output features
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        # Initialize weights and biases directly for better control
        self.weight1 = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.bias1 = nn.Parameter(torch.Tensor(hidden_size))
        
        # Initialize parameters using the same method as nn.Linear
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias1, -bound, bound)
        
        # Store the transposed weight for more efficient matrix multiplication
        self.register_buffer('weight1_t', self.weight1.t().contiguous())
        self._weight1_version = self.weight1._version
        
        # Keep the second linear layer for compatibility
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Ensure input is contiguous for optimal memory access
        x = x.contiguous() if not x.is_contiguous() else x
        
        # Update the transposed weight if the weight has changed
        if self._weight1_version != self.weight1._version:
            with torch.no_grad():
                self.weight1_t.copy_(self.weight1.t().contiguous())
            self._weight1_version = self.weight1._version
        
        # Fused matrix multiplication and bias addition using addmm
        # This is more efficient than separate matmul and add operations
        hidden = torch.addmm(self.bias1, x, self.weight1_t)
        
        # Apply sigmoid activation in-place to reduce memory allocations
        hidden.sigmoid_()
        
        # Sum across hidden dimension (dim=1) 
        summed = hidden.sum(dim=1)
        
        # Highly optimized logsumexp implementation with numerical stability
        max_val = summed.max()
        summed.sub_(max_val)  # In-place subtraction
        summed.exp_()         # In-place exponential
        sum_exp = summed.sum()
        result = max_val + sum_exp.log()
        
        return result

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_size = 10
hidden_size = 20
output_size = 5

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [input_size, hidden_size, output_size]