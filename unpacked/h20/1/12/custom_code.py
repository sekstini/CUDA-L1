import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized implementation for diagonal matrix multiplication.
    C = diag(A) * B
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication using optimized broadcasting.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        # Use unsqueeze(1) for efficient broadcasting
        # This reshapes A from (N,) to (N,1) for broadcasting with B
        return A.unsqueeze(1) * B

# Keep hyperparameters exactly as in the reference implementation
M = 4096
N = 4096

def get_inputs():
    A = torch.randn(N)
    B = torch.randn(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed