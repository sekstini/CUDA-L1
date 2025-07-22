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
        # Create a standard linear layer for proper initialization
        linear = nn.Linear(input_size, hidden_size)
        
        # Pre-transpose the weight matrix for optimal performance with addmm
        self.weight_t = nn.Parameter(linear.weight.t().contiguous())
        self.bias = nn.Parameter(linear.bias.clone())
        
        # Store parameters
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        # Pre-compute doubled scale factor for optimization
        self.doubled_scale_factor = scale_factor * 2.0
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size)
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Linear transformation using addmm for better CUDA performance
        # addmm: out = beta * input + alpha * (mat1 @ mat2)
        # Here beta=1, alpha=1, input=bias, mat1=x, mat2=weight_t
        out = torch.addmm(self.bias, x, self.weight_t)
        
        # Scale and add residual connection in one operation
        # This is equivalent to: out = out * scale_factor + out * scale_factor
        out.mul_(self.doubled_scale_factor)
        
        # Apply clamping in-place to avoid creating a new tensor
        out.clamp_(self.clamp_min, self.clamp_max)
        
        # Optimized logsumexp implementation
        # First find the maximum value for numerical stability using amax (faster than max)
        max_val = torch.amax(out, dim=1, keepdim=True)
        
        # Compute exp(x - max_val) and sum, then take log and add max_val back
        # This is mathematically equivalent to logsumexp but more efficient
        # Modify out in-place to avoid creating new tensors
        out.sub_(max_val)  # In-place subtraction: out = out - max_val
        out.exp_()         # In-place exponential: out = exp(out)
        out = torch.sum(out, dim=1, keepdim=True)
        out.log_()         # In-place logarithm: out = log(out)
        out.add_(max_val)  # In-place addition: out = out + max_val
        
        # Apply mish activation and multiply in-place
        # Mish(x) = x * tanh(softplus(x))
        mish_out = F.mish(out)
        out.mul_(mish_out)  # In-place multiplication
        
        return out

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