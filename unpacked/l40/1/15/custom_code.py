import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cuda_kernel = None
        if torch.cuda.is_available():
            self._init_cuda_kernel()
    
    def _init_cuda_kernel(self):
        cuda_kernel_code = """
        extern "C" __global__ void tril_matmul_kernel(
            const float* __restrict__ A,
            const float* __restrict__ B,
            float* __restrict__ C,
            const int N)
        {
            // Block indices
            const int block_row = blockIdx.y;
            const int block_col = blockIdx.x;
            
            // Block size
            const int BLOCK_SIZE = 32;
            
            // Skip blocks that are entirely in the upper triangular region
            // More precise condition: if the lowest row in this block is less than the lowest column
            if ((block_row * BLOCK_SIZE) < (block_col * BLOCK_SIZE)) {
                return;
            }
            
            // Thread indices
            const int thread_row = threadIdx.y;
            const int thread_col = threadIdx.x;
            
            // Global row and column indices
            const int row = block_row * BLOCK_SIZE + thread_row;
            const int col = block_col * BLOCK_SIZE + thread_col;
            
            // Early exit if outside matrix bounds or upper triangular region
            // Combined check to minimize divergence
            if (row >= N || col >= N || row < col) {
                return;
            }
            
            // Pre-compute row offset for memory access
            const int row_offset = row * N;
            
            // Shared memory for tiles of A and B with minimal padding to avoid bank conflicts
            __shared__ float As[32][33];
            __shared__ float Bs[32][33];
            
            // Register for accumulating result
            float sum = 0.0f;
            
            // For triangular matrices, we only need k where col <= k <= row
            const int k_start = col;
            const int k_end = min(row + 1, N);
            
            // Calculate number of tiles needed
            const int num_tiles = (k_end - k_start + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
            // Process tiles
            for (int t = 0; t < num_tiles; ++t) {
                // Starting k index for this tile
                const int tile_k_start = k_start + t * BLOCK_SIZE;
                const int tile_k_end = min(tile_k_start + BLOCK_SIZE, k_end);
                
                // Load tile of A into shared memory - coalesced access
                // Only load elements that are needed for computation
                if (tile_k_start + thread_col < tile_k_end) {
                    As[thread_row][thread_col] = A[row_offset + (tile_k_start + thread_col)];
                } else {
                    As[thread_row][thread_col] = 0.0f;
                }
                
                // Load tile of B into shared memory - coalesced access
                // Only load elements that are needed for computation
                if (tile_k_start + thread_row < tile_k_end) {
                    Bs[thread_row][thread_col] = B[(tile_k_start + thread_row) * N + col];
                } else {
                    Bs[thread_row][thread_col] = 0.0f;
                }
                
                // Synchronize to make sure the tiles are loaded
                __syncthreads();
                
                // Compute partial dot product for this tile with optimized loop unrolling
                // Only iterate up to the actual number of elements in the tile
                const int k_limit = min(BLOCK_SIZE, tile_k_end - tile_k_start);
                
                #pragma unroll 8
                for (int k = 0; k < k_limit; ++k) {
                    sum += As[thread_row][k] * Bs[k][thread_col];
                }
                
                // Synchronize before loading the next tile
                __syncthreads();
            }
            
            // Write result to global memory
            C[row_offset + col] = sum;
        }
        """
        
        try:
            from torch.utils.cpp_extension import load_inline
            self.cuda_kernel = load_inline(
                name="tril_matmul_cuda",
                cpp_sources="",
                cuda_sources=cuda_kernel_code,
                functions=["tril_matmul_kernel"],
                with_cuda=True,
                verbose=False
            )
        except Exception as e:
            print(f"Failed to load CUDA kernel: {e}")
            self.cuda_kernel = None
    
    def forward(self, A, B):
        """
        Performs optimized matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        N = A.shape[0]
        device = A.device
        dtype = A.dtype
        
        # For non-CUDA tensors or when CUDA kernel failed to load, use optimized PyTorch implementation
        if self.cuda_kernel is None or not torch.cuda.is_available() or not A.is_cuda:
            return self._forward_pytorch(A, B)
        
        # Use our custom CUDA kernel
        C = torch.zeros((N, N), dtype=dtype, device=device)
        
        # Ensure contiguous tensors for optimal memory access
        A = A.contiguous()
        B = B.contiguous()
        
        # Determine grid and block dimensions
        block_size = 32
        grid_x = math.ceil(N / block_size)
        grid_y = math.ceil(N / block_size)
        
        # Launch the kernel
        self.cuda_kernel.tril_matmul_kernel(
            grid=(grid_x, grid_y),
            block=(block_size, block_size),
            args=[A.data_ptr(), B.data_ptr(), C.data_ptr(), N],
        )
        
        return C
    
    def _forward_pytorch(self, A, B):
        """
        Optimized PyTorch implementation for when CUDA kernel is not available.
        This implementation also exploits the triangular structure to reduce computation.
        """
        N = A.shape[0]
        device = A.device
        dtype = A.dtype
        
        # For very small matrices, use built-in operations
        if N <= 128:
            return torch.tril(torch.matmul(A, B))
        
        # Pre-allocate result matrix
        C = torch.zeros((N, N), dtype=dtype, device=device)
        
        # Adaptive block size based on matrix size
        if N <= 1024:
            block_size = 256
        elif N <= 2048:
            block_size = 512
        else:
            block_size = 1024
        
        # Optimized triangular matrix multiplication using block-based approach
        for i in range(0, N, block_size):
            i_end = min(i + block_size, N)
            
            # Process only the lower triangular blocks
            for j in range(0, i_end, block_size):
                j_end = min(j + block_size, N)
                
                # For this output block C[i:i_end, j:j_end], we only need to compute
                # sum over k of A[i:i_end, k] * B[k, j:j_end]
                # But we only need k from j to i_end due to triangular structure
                k_start = j
                k_end = i_end
                
                if k_start < k_end:
                    # Extract the relevant portions of A and B for this computation
                    A_slice = A[i:i_end, k_start:k_end]
                    B_slice = B[k_start:k_end, j:j_end]
                    
                    # Perform the matrix multiplication for this block
                    C[i:i_end, j:j_end] = torch.matmul(A_slice, B_slice)
        
        return C

M = 4096

def get_inputs():
    A = torch.randn(M, M)
    B = torch.randn(M, M)
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed