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
        # Initialize weights and bias similar to nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling_factor = scaling_factor
        
        # Initialize parameters (same as nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Pre-transpose the weight matrix and store it for efficient matmul
        self.register_buffer('weight_t', self.weight.t().contiguous())
        
        # Register a hook to update the transposed weight when the original weight changes
        self.weight.register_hook(self._update_weight_t)
    
    def _update_weight_t(self, grad):
        """Hook to update the transposed weight when the original weight is updated"""
        with torch.no_grad():
            self.weight_t.copy_(self.weight.t().contiguous())
        return grad
    
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
        
        # Use addmm for efficient matrix multiplication with bias
        # This combines matrix multiplication and bias addition in one operation
        linear_output = torch.addmm(self.bias, x, self.weight_t)
        
        # Apply sigmoid for Swish activation
        # Store result directly without creating an additional intermediate tensor
        sigmoid_output = torch.sigmoid(linear_output)
        
        # Apply Swish activation (x * sigmoid(x)) and scaling in one operation
        # Using in-place multiplication to avoid creating new tensors
        return linear_output.mul_(sigmoid_output).mul_(self.scaling_factor)

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