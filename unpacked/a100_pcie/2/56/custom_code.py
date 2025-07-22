import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        # Initialize weights and bias directly instead of using nn.Linear
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()
        
        # Pre-transpose the weight matrix to avoid transposition in forward pass
        # Store as a non-trainable parameter for optimal performance
        self.weight_t = nn.Parameter(self.weight.t(), requires_grad=False)
        
    def reset_parameters(self):
        # Use the same initialization as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        # Use addmm for efficient fused matrix multiplication with bias addition
        linear_output = torch.addmm(self.bias, x, self.weight_t)
        
        # Apply sigmoid activation
        sigmoid_output = torch.sigmoid(linear_output)
        
        # Sum reduction along dimension 1 with keepdim=True
        result = torch.sum(sigmoid_output, dim=1, keepdim=True)
        
        return result

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_size = 10
hidden_size = 20

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [input_size, hidden_size]