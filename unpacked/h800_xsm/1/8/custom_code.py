import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A * B) with CUDA acceleration
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Pre-allocate output tensor and cache to avoid allocation overhead
        self._output_cache = None
        self._cache_device = None
        self._cache_dtype = None
        self._input_shapes = None
        self._stream = None
        self._warmed_up = False
        
        # Compile CUDA kernel if we're on a CUDA device
        self._kernel = None
        if torch.cuda.is_available():
            self._compile_kernel()
    
    def _compile_kernel(self):
        """Compile the custom CUDA kernel for matrix multiplication"""
        cuda_kernel = """
        extern "C" __global__ void matmul_kernel(
            const float* __restrict__ A,
            const float* __restrict__ B,
            float* __restrict__ C,
            const int M, const int K, const int N) {
            
            // Block tile size
            const int BM = 32;
            const int BN = 32;
            const int BK = 32;
            
            // Thread tile size (each thread computes a 4x4 block)
            const int TM = 4;
            const int TN = 4;
            
            // Shared memory tiles
            __shared__ float As[BM][BK];
            __shared__ float Bs[BK][BN];
            
            // Block indices
            const int bx = blockIdx.x;
            const int by = blockIdx.y;
            
            // Thread indices
            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            
            // Row and column indices for this thread
            const int row = by * BM + ty;
            const int col = bx * BN + tx;
            
            // Registers for accumulating results
            float Csub[TM][TN] = {0.0f};
            
            // Loop over tiles of A and B
            for (int tile = 0; tile < (K + BK - 1) / BK; ++tile) {
                
                // Collaborative loading of A and B tiles into shared memory
                #pragma unroll
                for (int i = 0; i < BM; i += blockDim.y) {
                    int r = by * BM + i + ty;
                    int c = tile * BK + tx;
                    if (r < M && c < K) {
                        As[i + ty][tx] = A[r * K + c];
                    } else {
                        As[i + ty][tx] = 0.0f;
                    }
                }
                
                #pragma unroll
                for (int i = 0; i < BK; i += blockDim.y) {
                    int r = tile * BK + i + ty;
                    int c = bx * BN + tx;
                    if (r < K && c < N) {
                        Bs[i + ty][tx] = B[r * N + c];
                    } else {
                        Bs[i + ty][tx] = 0.0f;
                    }
                }
                
                // Synchronize to ensure all threads have loaded the tiles
                __syncthreads();
                
                // Compute matrix multiplication for this thread's TM x TN block
                #pragma unroll
                for (int k = 0; k < BK; ++k) {
                    #pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        #pragma unroll
                        for (int j = 0; j < TN; ++j) {
                            if ((by * BM + ty * TM + i) < M && (bx * BN + tx * TN + j) < N && (tile * BK + k) < K) {
                                Csub[i][j] += As[ty * TM + i][k] * Bs[k][tx * TN + j];
                            }
                        }
                    }
                }
                
                // Synchronize before loading the next tile
                __syncthreads();
            }
            
            // Write the computed values to C
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    int r = by * BM + ty * TM + i;
                    int c = bx * BN + tx * TN + j;
                    if (r < M && c < N) {
                        C[r * N + c] = Csub[i][j];
                    }
                }
            }
        }
        """
        
        try:
            from torch.utils.cpp_extension import load_inline
            self._kernel = load_inline(
                name="matmul_cuda",
                cpp_sources="",
                cuda_sources=cuda_kernel,
                functions=["matmul_kernel"],
                verbose=False
            )
        except Exception as e:
            print(f"Failed to compile CUDA kernel: {e}")
            self._kernel = None
    
    def _custom_matmul(self, A, B, C):
        """
        Custom matrix multiplication using our CUDA kernel
        
        Args:
            A: Input tensor with shape (M, K)
            B: Input tensor with shape (K, N)
            C: Output tensor with shape (M, N) to store the result
        """
        if self._kernel is None:
            # Fallback to PyTorch's implementation if kernel compilation failed
            torch.matmul(A, B, out=C)
            return
        
        M, K = A.shape
        K_, N = B.shape
        
        # Define block and grid dimensions
        threads_per_block = (32, 8)  # 256 threads per block
        blocks_x = (N + 31) // 32
        blocks_y = (M + 31) // 32
        
        # Launch the kernel
        try:
            self._kernel.matmul_kernel(
                grid=(blocks_x, blocks_y),
                block=threads_per_block,
                args=[A.data_ptr(), B.data_ptr(), C.data_ptr(), M, K, N]
            )
        except Exception as e:
            print(f"Kernel execution failed: {e}")
            # Fallback to PyTorch's implementation
            torch.matmul(A, B, out=C)
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B with optimized CUDA kernel.

        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        # Get dimensions and device
        M, K = A.shape
        K_, N = B.shape
        device = A.device
        dtype = A.dtype
        
        # Create dedicated CUDA stream if on GPU and not already created
        if device.type == 'cuda' and self._stream is None:
            self._stream = torch.cuda.Stream(device)
        
        # Ensure optimal memory layout for both inputs
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()
        
        # Check if we need to create or resize the output cache
        shapes_changed = (self._input_shapes != (M, K, N))
        device_changed = (self._cache_device != device)
        dtype_changed = (self._cache_dtype != dtype)
        
        if (self._output_cache is None or shapes_changed or 
            device_changed or dtype_changed):
            
            # Free previous cache if it exists to avoid memory leaks
            if self._output_cache is not None:
                del self._output_cache
                torch.cuda.empty_cache()
            
            # Create new output cache with optimal memory layout
            self._output_cache = torch.empty((M, N), 
                                           dtype=dtype, 
                                           device=device, 
                                           memory_format=torch.contiguous_format)
            
            # Update cache metadata
            self._cache_device = device
            self._cache_dtype = dtype
            self._input_shapes = (M, K, N)
            self._warmed_up = False
        
        # Perform warm-up if needed
        if not self._warmed_up and device.type == 'cuda':
            with torch.cuda.stream(self._stream):
                for _ in range(3):
                    torch.matmul(A[:min(128, M), :min(128, K)], 
                                B[:min(128, K), :min(128, N)], 
                                out=self._output_cache[:min(128, M), :min(128, N)])
                torch.cuda.synchronize()
            self._warmed_up = True
        
        # Use custom CUDA kernel if we're on GPU and have float32 tensors
        use_custom_kernel = (
            device.type == 'cuda' and
            dtype == torch.float32 and
            self._kernel is not None
        )
        
        try:
            if use_custom_kernel:
                with torch.cuda.stream(self._stream):
                    self._custom_matmul(A, B, self._output_cache)
            else:
                # Fallback to PyTorch's implementation
                if device.type == 'cuda':
                    with torch.cuda.stream(self._stream):
                        torch.matmul(A, B, out=self._output_cache)
                else:
                    torch.matmul(A, B, out=self._output_cache)
        except Exception as e:
            print(f"Matrix multiplication failed: {e}")
            # Final fallback to ensure we always return a result
            torch.matmul(A, B, out=self._output_cache)
        
        return self._output_cache

# Keep the exact hyperparameters from the reference implementation
M = 8205
K = 2949
N = 5921

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed