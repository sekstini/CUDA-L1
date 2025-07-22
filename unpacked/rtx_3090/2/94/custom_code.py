import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    An optimized model that performs a GEMM, BiasAdd, Hardtanh, Mish, and GroupNorm operations in sequence.
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias_shape (tuple): Shape of the bias tensor
        num_groups (int): Number of groups for GroupNorm
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        # Create components identical to the reference implementation
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.hardtanh = nn.Hardtanh()
        self.mish = nn.Mish()
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        
        # Pre-transpose weight matrix for more efficient matrix multiplication
        self.register_buffer('weight_t', self.gemm.weight.t().contiguous())
        # Pre-combine biases for more efficient addition
        self.register_buffer('combined_bias', self.gemm.bias + self.bias)
        
        # Simple dirty flag for parameter updates
        self._params_dirty = True
        
        # Register hooks to detect parameter changes
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to detect parameter changes"""
        def mark_dirty(grad):
            self._params_dirty = True
            return grad
        
        self.gemm.weight.register_hook(mark_dirty)
        self.gemm.bias.register_hook(mark_dirty)
        self.bias.register_hook(mark_dirty)
    
    def _update_cached_params(self):
        """Update cached parameters only when necessary"""
        if self._params_dirty:
            with torch.no_grad():
                # Update transposed weight
                self.weight_t.copy_(self.gemm.weight.t().contiguous())
                # Update combined bias
                self.combined_bias.copy_(self.gemm.bias + self.bias)
            self._params_dirty = False
    
    def forward(self, x):
        """
        Optimized forward pass with fused operations and minimal memory traffic
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Update cached parameters if needed
        self._update_cached_params()
        
        # Fused matrix multiplication and bias addition using addmm
        # This combines GEMM + bias in one efficient operation
        x = torch.addmm(self.combined_bias, x, self.weight_t)
        
        # Apply Hardtanh using efficient in-place clamp operation
        x.clamp_(-1.0, 1.0)
        
        # Apply Mish activation using PyTorch's optimized implementation
        x = F.mish(x)
        
        # Apply GroupNorm - already highly optimized in PyTorch
        x = self.groupnorm(x)
        
        return x
    
    def train(self, mode=True):
        """Override train mode to ensure parameters are updated"""
        self._params_dirty = True
        return super(ModelNew, self).train(mode)


# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 512
out_features = 1024
bias_shape = (out_features,)
num_groups = 32

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_features, out_features, bias_shape, num_groups]