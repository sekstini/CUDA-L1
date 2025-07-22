import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Simple model that performs matrix multiplication (C = A * B) for upper triangular matrices.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        N = A.shape[0]
        
        # For small matrices, use the native implementation
        if N <= 1024:
            return torch.triu(torch.matmul(A, B))
        
        # Ensure tensors are contiguous for optimal performance
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()
        
        # Use block size of 512 (optimal from previous attempts)
        block_size = 512
        num_blocks = (N + block_size - 1) // block_size
        
        # Pre-allocate output matrix
        C = torch.zeros_like(A, memory_format=torch.contiguous_format)
        
        if A.is_cuda:
            # Use 4 CUDA streams (optimal from previous attempts)
            num_streams = 4
            streams = [torch.cuda.Stream() for _ in range(num_streams)]
            current_stream = torch.cuda.current_stream()
            
            # Create a list of block pairs to process
            block_pairs = []
            for i in range(num_blocks):
                for j in range(i, num_blocks):
                    # Calculate approximate workload (higher for diagonal blocks)
                    workload = (j - i + 1) * block_size * block_size
                    block_pairs.append((i, j, workload))
            
            # Sort by workload in descending order for better load balancing
            block_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Process blocks with optimized scheduling
            for idx, (i, j, _) in enumerate(block_pairs):
                i_start = i * block_size
                i_end = min((i + 1) * block_size, N)
                j_start = j * block_size
                j_end = min((j + 1) * block_size, N)
                
                # Use round-robin stream assignment
                stream_idx = idx % num_streams
                with torch.cuda.stream(streams[stream_idx]):
                    if i == j:
                        # Diagonal block - apply triu after multiplication
                        A_block = A[i_start:i_end, i_start:i_end]
                        B_block = B[i_start:i_end, i_start:i_end]
                        C_block = torch.matmul(A_block, B_block)
                        C[i_start:i_end, j_start:j_end] = torch.triu(C_block)
                    else:
                        # Off-diagonal block - direct computation
                        # Extract only the necessary parts of A and B
                        A_block = A[i_start:i_end, i_start:j_end]
                        B_block = B[i_start:j_end, j_start:j_end]
                        C[i_start:i_end, j_start:j_end] = torch.matmul(A_block, B_block)
            
            # Synchronize all streams
            for stream in streams:
                current_stream.wait_stream(stream)
        else:
            # CPU implementation
            for i in range(num_blocks):
                i_start = i * block_size
                i_end = min((i + 1) * block_size, N)
                
                for j in range(i, num_blocks):
                    j_start = j * block_size
                    j_end = min((j + 1) * block_size, N)
                    
                    if i == j:
                        # Diagonal block
                        A_block = A[i_start:i_end, i_start:i_end]
                        B_block = B[i_start:i_end, i_start:i_end]
                        C_block = torch.matmul(A_block, B_block)
                        C[i_start:i_end, j_start:j_end] = torch.triu(C_block)
                    else:
                        # Off-diagonal block
                        A_block = A[i_start:i_end, i_start:j_end]
                        B_block = B[i_start:j_end, j_start:j_end]
                        C[i_start:i_end, j_start:j_end] = torch.matmul(A_block, B_block)
        
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