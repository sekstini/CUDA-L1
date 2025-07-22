import torch
import torch.nn as nn
import math

# CUDA kernel for optimized matrix multiplication
cuda_kernel_code = """
extern "C" __global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int K, const int N) {
    
    // Block dimensions
    constexpr int BLOCK_SIZE_M = 32;
    constexpr int BLOCK_SIZE_N = 32;
    constexpr int BLOCK_SIZE_K = 32;
    
    // Thread coarsening factors
    constexpr int THREAD_SIZE_M = 4;
    constexpr int THREAD_SIZE_N = 4;
    
    // Shared memory for tiles
    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    
    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // First block in the block-row of C
    const int block_row_start = by * BLOCK_SIZE_M;
    const int block_col_start = bx * BLOCK_SIZE_N;
    
    // Initialize accumulator registers
    float accum[THREAD_SIZE_M][THREAD_SIZE_N] = {0.0f};
    
    // Loop over all tiles
    const int num_tiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
    for (int tile = 0; tile < num_tiles; ++tile) {
        // Load A tile into shared memory
        for (int m = 0; m < THREAD_SIZE_M; ++m) {
            for (int k = 0; k < THREAD_SIZE_N; ++k) {
                const int m_idx = block_row_start + ty * THREAD_SIZE_M + m;
                const int k_idx = tile * BLOCK_SIZE_K + tx * THREAD_SIZE_N + k;
                if (m_idx < M && k_idx < K) {
                    As[ty * THREAD_SIZE_M + m][tx * THREAD_SIZE_N + k] = A[m_idx * K + k_idx];
                } else {
                    As[ty * THREAD_SIZE_M + m][tx * THREAD_SIZE_N + k] = 0.0f;
                }
            }
        }
        
        // Load B tile into shared memory (transposed access for B.T)
        for (int k = 0; k < THREAD_SIZE_M; ++k) {
            for (int n = 0; n < THREAD_SIZE_N; ++n) {
                const int k_idx = tile * BLOCK_SIZE_K + ty * THREAD_SIZE_M + k;
                const int n_idx = block_col_start + tx * THREAD_SIZE_N + n;
                if (k_idx < K && n_idx < N) {
                    // Access B in transposed manner (B.T)
                    Bs[ty * THREAD_SIZE_M + k][tx * THREAD_SIZE_N + n] = B[n_idx * K + k_idx];
                } else {
                    Bs[ty * THREAD_SIZE_M + k][tx * THREAD_SIZE_N + n] = 0.0f;
                }
            }
        }
        
        // Synchronize to ensure all threads have loaded the tiles
        __syncthreads();
        
        // Compute matrix multiplication for this tile
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            for (int m = 0; m < THREAD_SIZE_M; ++m) {
                for (int n = 0; n < THREAD_SIZE_N; ++n) {
                    accum[m][n] += As[ty * THREAD_SIZE_M + m][k] * Bs[k][tx * THREAD_SIZE_N + n];
                }
            }
        }
        
        // Synchronize before loading the next tile
        __syncthreads();
    }
    
    // Store results back to global memory
    for (int m = 0; m < THREAD_SIZE_M; ++m) {
        for (int n = 0; n < THREAD_SIZE_N; ++n) {
            const int m_idx = block_row_start + ty * THREAD_SIZE_M + m;
            const int n_idx = block_col_start + tx * THREAD_SIZE_N + n;
            if (m_idx < M && n_idx < N) {
                C[m_idx * N + n_idx] = accum[m][n];
            }
        }
    }
}
"""

class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Compile the CUDA kernel if CUDA is available
        if torch.cuda.is_available():
            self.matmul_kernel = None
            try:
                self.matmul_kernel = torch.cuda.compile_ptx(cuda_kernel_code, options=[
                    "-use_fast_math",
                    "-restrict",
                    "-lineinfo"
                ])
            except Exception as e:
                print(f"Warning: Failed to compile CUDA kernel: {e}")
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        # Move tensors to CUDA if available
        if torch.cuda.is_available():
            if not A.is_cuda:
                A = A.cuda(non_blocking=True)
            if not B.is_cuda:
                B = B.cuda(non_blocking=True)
            
            # Ensure tensors are contiguous for optimal memory access
            A = A if A.is_contiguous() else A.contiguous()
            B = B if B.is_contiguous() else B.contiguous()
            
            # Get tensor dimensions
            M, K = A.shape
            N, K_check = B.shape
            
            assert K == K_check, "Inner dimensions must match for matrix multiplication"
            
            # Use the custom CUDA kernel if available
            if self.matmul_kernel is not None:
                try:
                    with torch.cuda.stream(self.stream):
                        # Allocate output tensor
                        C = torch.empty((M, N), dtype=A.dtype, device=A.device)
                        
                        # Configure kernel launch parameters
                        threads_per_block = (32, 8)  # 256 threads per block
                        blocks_per_grid = (
                            math.ceil(N / (32 * 4)),  # Account for thread coarsening
                            math.ceil(M / (32 * 4))   # Account for thread coarsening
                        )
                        
                        # Launch the kernel
                        torch.cuda.launch_kernel(
                            self.matmul_kernel,
                            blocks_per_grid,
                            threads_per_block,
                            [A.data_ptr(), B.data_ptr(), C.data_ptr(), M, K, N]
                        )
                        
                        return C
                except Exception as e:
                    # Fallback to PyTorch implementation if kernel launch fails
                    print(f"Warning: Custom kernel failed, falling back to PyTorch: {e}")
            
            # Fallback to PyTorch's optimized implementation
            with torch.cuda.stream(self.stream):
                return torch.matmul(A, B.T)
        else:
            # CPU implementation
            A = A if A.is_contiguous() else A.contiguous()
            B = B if B.is_contiguous() else B.contiguous()
            return torch.matmul(A, B.T)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
M = 1024
K = 4096
N = 2048

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(N, K)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed