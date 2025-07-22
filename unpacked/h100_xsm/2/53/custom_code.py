import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelNew(nn.Module):
    """
    Your optimized implementation here that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features  
        scaling_factor (float): Scaling factor to apply
        hardtanh_min (float): Minimum value for hardtanh
        hardtanh_max (float): Maximum value for hardtanh
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        
        # Create weight and bias parameters with optimal memory layout
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
        
        # Initialize parameters using the same approach as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Pre-scale the weights and bias by scaling_factor to save one multiplication
        # Use register_buffer to ensure these are moved to GPU with the model
        self.register_buffer('scaled_weight', self.weight.data * scaling_factor)
        self.register_buffer('scaled_bias', self.bias.data * scaling_factor)
        
        # Create optimized memory layout for weights
        self.register_buffer('weight_t', self.weight.data.t().contiguous())
        self.register_buffer('scaled_weight_t', self.scaled_weight.data.t().contiguous())
    
    def _update_scaled_params(self):
        """Update scaled weights and biases when original parameters change"""
        with torch.no_grad():
            self.scaled_weight.copy_(self.weight * self.scaling_factor)
            self.scaled_bias.copy_(self.bias * self.scaling_factor)
            self.weight_t.copy_(self.weight.t().contiguous())
            self.scaled_weight_t.copy_(self.scaled_weight.t().contiguous())
    
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
        
        # Use optimized matrix multiplication with pre-scaled weights
        # This eliminates one multiplication operation across the entire tensor
        # Use the transposed weight for better memory access patterns
        out = torch.matmul(x, self.scaled_weight_t)
        
        # Add bias
        out = out + self.scaled_bias
        
        # Apply hardtanh (clamping)
        out = torch.clamp(out, self.hardtanh_min, self.hardtanh_max)
        
        # Apply GELU activation with the most optimized version
        out = F.gelu(out)
        
        return out
    
    def _apply(self, fn):
        """Override to keep scaled weights in sync when model is moved between devices"""
        result = super(ModelNew, self)._apply(fn)
        # Update scaled parameters after applying function (e.g., .cuda(), .cpu())
        if hasattr(self, 'scaled_weight') and hasattr(self, 'scaled_bias'):
            self._update_scaled_params()
        return result
    
    def train(self, mode=True):
        """Override train mode to ensure scaled parameters stay in sync"""
        result = super(ModelNew, self).train(mode)
        if mode:
            self._update_scaled_params()
        return result

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation  
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]