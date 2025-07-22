import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A * B) for upper triangular matrices.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs optimized matrix multiplication for upper triangular matrices.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        N = A.shape[0]
        device = A.device
        dtype = A.dtype
        
        # Pre-allocate output matrix with zeros
        C = torch.zeros((N, N), dtype=dtype, device=device)
        
        # Optimal block size based on previous experiments
        block_size = 768
        
        # Process only upper triangular blocks (j >= i)
        for i in range(0, N, block_size):
            i_end = min(i + block_size, N)
            i_slice = slice(i, i_end)
            
            for j in range(i, N, block_size):
                j_end = min(j + block_size, N)
                j_slice = slice(j, j_end)
                
                # For upper triangular matrices, we only need to sum over k from i to j_end
                # Since A[i:i_end, k<i] = 0 and B[k>j_end, j:j_end] = 0
                k_slice = slice(i, j_end)
                
                # Extract the relevant blocks from A and B
                A_block = A[i_slice, k_slice]
                B_block = B[k_slice, j_slice]
                
                # Use beta=0 for the operation to avoid adding to zeros
                # This is equivalent to C[i:i_end, j:j_end] = A_block @ B_block
                torch.addmm(
                    input=C[i_slice, j_slice],
                    mat1=A_block,
                    mat2=B_block,
                    beta=0.0,  # Override the zeros instead of adding to them
                    alpha=1.0,
                    out=C[i_slice, j_slice]
                )
        
        return C

N = 4096

def get_inputs():
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: A list containing two upper triangular matrices of shape (N, N).
    """
    A = torch.triu(torch.randn(N, N))
    B = torch.triu(torch.randn(N, N))
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs are needed for this model.

    Returns:
        list: An empty list.
    """
    return []