import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix-scalar multiplication (C = A * s)
    using buffer reuse and CUDA stream optimization.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.output_buffer = None
        self.buffer_shape = None
        self.buffer_device = None
        self.buffer_dtype = None
        self.cuda_stream = None
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        # Ensure input is contiguous for optimal memory access
        if not A.is_contiguous():
            A = A.contiguous()
        
        # Initialize CUDA stream if on GPU and not already created
        if A.is_cuda and self.cuda_stream is None:
            self.cuda_stream = torch.cuda.Stream(priority=-1)  # High priority stream
        
        # Check if we need to create or resize the output buffer
        if (self.output_buffer is None or 
            self.buffer_shape != A.shape or 
            self.buffer_device != A.device or 
            self.buffer_dtype != A.dtype):
            
            # Update tracking variables
            self.buffer_shape = A.shape
            self.buffer_device = A.device
            self.buffer_dtype = A.dtype
            
            # Create new buffer with same properties as input
            self.output_buffer = torch.empty_like(A)
        
        # Use the dedicated stream if on CUDA
        if A.is_cuda and self.cuda_stream is not None:
            with torch.cuda.stream(self.cuda_stream):
                # Use out parameter for multiplication to avoid allocating new memory
                torch.mul(A, s, out=self.output_buffer)
        else:
            # Use out parameter for multiplication to avoid allocating new memory
            torch.mul(A, s, out=self.output_buffer)
        
        return self.output_buffer

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
M = 16384
N = 4096

def get_inputs():
    A = torch.randn(M, N)
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []  # No special initialization inputs needed