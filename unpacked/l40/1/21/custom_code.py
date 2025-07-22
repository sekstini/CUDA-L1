import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs a Sigmoid activation using in-place operations
    to eliminate memory allocation overhead.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid activation to the input tensor using in-place operations
        for maximum performance.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        # Use in-place sigmoid operation to eliminate memory allocation overhead
        torch.sigmoid_(x)
        return x

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed