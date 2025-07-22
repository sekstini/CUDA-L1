import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs a Softsign activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Cache tensors to avoid repeated allocations
        self._cached_denom = None
        self._cached_output = None
        self._cached_shape = None
        self._cached_device = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softsign activation to the input tensor with optimizations.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softsign applied, same shape as input.
        """
        # Ensure contiguous memory layout for optimal performance
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Check if we can reuse cached tensors
        if (self._cached_output is not None and 
            self._cached_shape == x.shape and 
            self._cached_device == x.device):
            
            # Compute abs(x) directly into cached denominator tensor
            torch.abs(x, out=self._cached_denom)
            
            # Add 1 to denominator in-place
            self._cached_denom.add_(1.0)
            
            # Divide x by denominator directly into the result tensor
            torch.div(x, self._cached_denom, out=self._cached_output)
            
            return self._cached_output
        else:
            # Create new cached tensors
            self._cached_denom = torch.empty_like(x)
            self._cached_output = torch.empty_like(x)
            self._cached_shape = x.shape
            self._cached_device = x.device
            
            # Compute abs(x) directly into cached denominator tensor
            torch.abs(x, out=self._cached_denom)
            
            # Add 1 to denominator in-place
            self._cached_denom.add_(1.0)
            
            # Divide x by denominator directly into the result tensor
            torch.div(x, self._cached_denom, out=self._cached_output)
            
            return self._cached_output

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed