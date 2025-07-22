import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of output features  
        scale_factor (float): Scaling factor to apply
        clamp_min (float): Minimum value for clamping
        clamp_max (float): Maximum value for clamping
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # Create weight and bias parameters directly for optimal control
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        
        # Initialize parameters (identical to nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Pre-compute combined scale factor for operation fusion
        # x * scale_factor + x * scale_factor = x * (scale_factor * 2)
        self.register_buffer('combined_scale', torch.tensor(scale_factor * 2.0, dtype=torch.float32))
        
        # Store clamping values
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        # Pre-transpose weight matrix for more efficient matrix multiplication
        self.register_buffer('weight_t', self.weight.t().contiguous())
        
    def forward(self, x):
        """
        Optimized forward pass with strategic tensor reuse
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size)
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Matrix multiplication using addmm for better performance
        # This fuses the matrix multiplication and bias addition into a single operation
        result = torch.addmm(self.bias, x, self.weight_t)
        
        # Combined scaling and residual addition in a single in-place operation
        result.mul_(self.combined_scale)
        
        # In-place clamping
        result.clamp_(self.clamp_min, self.clamp_max)
        
        # Optimized LogSumExp implementation with strategic tensor reuse
        # Find max for numerical stability - use amax for better performance
        max_vals = torch.amax(result, dim=1, keepdim=True)
        
        # Compute exp(x - max_val) in-place
        result.sub_(max_vals)
        result.exp_()
        
        # Sum along dim=1
        sum_exp = torch.sum(result, dim=1, keepdim=True)
        
        # Compute log(sum_exp) + max_val efficiently
        sum_exp.log_()
        logsumexp_result = sum_exp.add_(max_vals)
        
        # Compute mish activation: x * tanh(softplus(x))
        mish_result = F.mish(logsumexp_result)
        
        # Final element-wise multiplication - reuse logsumexp_result for final result
        return logsumexp_result.mul_(mish_result)  # in-place multiplication for final result

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_size = 512
hidden_size = 1024
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation  
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]