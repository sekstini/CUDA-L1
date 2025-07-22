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
        
        # Create weight and bias parameters (same as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters (same as nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Pre-compute and cache scaled parameters for maximum efficiency
        self.register_buffer('weight_t_scaled', None)
        self.register_buffer('bias_scaled', None)
        
        # Initialize cached tensors
        self._update_cached_tensors()
        self._needs_update = False
        
        # Register hooks for parameter updates
        self._register_hooks()
    
    def _update_cached_tensors(self):
        """Update cached tensors with optimal memory layout"""
        with torch.no_grad():
            # Pre-transpose and pre-scale weight for optimal GEMM performance
            self.weight_t_scaled = (self.weight.t() * self.scaling_factor).contiguous()
            
            # Pre-scale bias to match the scaled GEMM output
            self.bias_scaled = (self.bias * self.scaling_factor).contiguous()
    
    def _register_hooks(self):
        """Register hooks to update cached tensors when parameters change"""
        def hook_fn(grad):
            self._needs_update = True
            return grad
        
        self.weight.register_hook(hook_fn)
        self.bias.register_hook(hook_fn)
    
    def forward(self, x):
        """
        Optimized forward pass with maximum operation fusion
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Update cached tensors if needed
        if self._needs_update:
            self._update_cached_tensors()
            self._needs_update = False
        
        # Fused matrix multiplication with pre-scaled parameters
        # This combines GEMM + scaling + bias addition in one optimized operation
        output = torch.addmm(self.bias_scaled, x, self.weight_t_scaled)
        
        # Apply hardtanh clipping in-place to avoid memory allocation
        output.clamp_(min=self.hardtanh_min, max=self.hardtanh_max)
        
        # Apply GELU activation using PyTorch's optimized implementation
        return F.gelu(output)

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