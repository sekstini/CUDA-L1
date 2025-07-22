import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Performs 3D tensor-matrix multiplication with highly optimized implementation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.stream = None
        self.cached_output = None
        self.cached_shape = None
        self.cached_device = None
        self.cached_dtype = None
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        # Get dimensions
        N, M, K = A.shape
        L = B.shape[1]
        
        # Initialize CUDA stream if on GPU and not already created
        if A.is_cuda and self.stream is None:
            self.stream = torch.cuda.Stream()
        
        # Fast path: check if tensors are already in optimal memory layout
        A_optimal = A.is_contiguous()
        B_optimal = B.is_contiguous()
        
        # Only make contiguous if necessary to avoid unnecessary copies
        A_work = A if A_optimal else A.contiguous()
        B_work = B if B_optimal else B.contiguous()
        
        # Pre-allocate output tensor for better memory management
        output_shape = (N, M, L)
        if (self.cached_output is None or 
            self.cached_shape != output_shape or 
            self.cached_device != A.device or
            self.cached_dtype != A.dtype):
            # Create new output tensor with optimal memory layout
            self.cached_output = torch.empty(N * M, L, dtype=A.dtype, device=A.device)
            self.cached_shape = output_shape
            self.cached_device = A.device
            self.cached_dtype = A.dtype
        
        # Reshape A to combine batch and M dimensions for efficient 2D GEMM
        # Use view instead of reshape when possible to avoid memory copies
        A_reshaped = A_work.view(N * M, K)
        
        # Perform optimized matrix multiplication with pre-allocated output
        if A.is_cuda:
            with torch.cuda.stream(self.stream):
                torch.mm(A_reshaped, B_work, out=self.cached_output)
        else:
            torch.mm(A_reshaped, B_work, out=self.cached_output)
        
        # Reshape back to the original 3D structure using view for zero-copy
        return self.cached_output.view(N, M, L)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
N = 16
M = 1024
K = 2048
L = 768

def get_inputs():
    A = torch.randn(N, M, K)
    B = torch.randn(K, L)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed