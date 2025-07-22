import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized implementation of L2 normalization using PyTorch's most efficient operations.
    """
    def __init__(self):
        """
        Initializes the L2Norm layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor using optimized vector norm computation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        # Use torch.linalg.vector_norm which is more optimized than torch.norm for vector norms
        norm = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
        
        # Normalize by division - PyTorch will optimize this operation
        return x / norm

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []