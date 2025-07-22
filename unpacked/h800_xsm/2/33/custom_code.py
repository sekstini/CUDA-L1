import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        scale_shape (tuple): Shape of the scaling factor
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Keep the same modules as the reference implementation
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        
        # Pre-compute and cache optimized matrices with optimal memory layout
        self.register_buffer('scaled_weight_t', None)
        self.register_buffer('scaled_bias', None)
        
        # Ultra-efficient version tracking with minimal overhead
        self._cached_versions = None
        
        # Cache tensor properties to avoid repeated attribute lookups
        self.has_bias = self.gemm.bias is not None
        self._weight_ref = self.gemm.weight
        self._scale_ref = self.scale
        self._bias_ref = self.gemm.bias if self.has_bias else None
        
        # Initial update of cached matrices
        self._update_cached_matrices()
    
    def _update_cached_matrices(self):
        """Update cached matrices with ultra-minimal overhead"""
        # Get current versions using cached references
        current_versions = (
            self._weight_ref._version,
            self._scale_ref._version,
            -1 if not self.has_bias else self._bias_ref._version
        )
        
        # Ultra-fast version comparison
        if self._cached_versions != current_versions:
            # Pre-compute transposed and scaled weight matrix for optimal CUDA execution
            # This eliminates the need for separate scaling in the forward pass
            weight_t = self._weight_ref.t().contiguous()
            scale_expanded = self._scale_ref.view(1, -1)
            self.scaled_weight_t = torch.mul(weight_t, scale_expanded).contiguous()
            
            # Pre-compute scaled bias if present to eliminate scaling operation
            if self.has_bias:
                self.scaled_bias = torch.mul(self._bias_ref, self._scale_ref).contiguous()
            
            # Update version cache with single assignment
            self._cached_versions = current_versions
    
    def forward(self, x):
        """
        Ultra-optimized forward pass with maximum CUDA efficiency
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Ensure optimal memory layout for CUDA kernels
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Update cached matrices with minimal overhead check
        self._update_cached_matrices()
        
        # Fused GEMM + scaling operation using pre-scaled weights
        # This leverages the most optimized CUDA kernels available
        if self.has_bias:
            # Use torch.addmm for maximum efficiency - single fused CUDA kernel
            # This combines matrix multiplication and bias addition in one operation
            out = torch.addmm(self.scaled_bias, x, self.scaled_weight_t)
        else:
            # Direct matrix multiplication with pre-scaled weights
            out = torch.mm(x, self.scaled_weight_t)
        
        # Apply batch normalization using PyTorch's optimized implementation
        out = self.bn(out)
        
        return out

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 1024
out_features = 512
scale_shape = (out_features,)

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_features, out_features, scale_shape]