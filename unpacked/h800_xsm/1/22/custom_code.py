import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Simple model that performs a Tanh activation with ultra-optimized execution.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self._output = None
        self._initialized = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation to the input tensor with optimized execution.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        # One-time initialization
        if not self._initialized:
            # Pre-allocate output tensor with optimal memory layout
            self._output = torch.empty_like(
                x, 
                memory_format=torch.contiguous_format
            )
            
            # Warm-up pass to ensure GPU kernels are cached
            with torch.no_grad():
                torch.tanh(x, out=self._output)
                
            self._initialized = True
            
            # Ultra-fast path - direct computation with pre-allocated output
            torch.tanh(x, out=self._output)
            return self._output
        
        # Ultra-fast path for all subsequent calls - direct computation with zero overhead
        torch.tanh(x, out=self._output)
        return self._output

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(batch_size, dim, device=device, memory_format=torch.contiguous_format)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed