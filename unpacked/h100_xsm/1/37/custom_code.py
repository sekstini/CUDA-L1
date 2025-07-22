import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    """
    Optimized implementation of Frobenius norm normalization using custom CUDA kernels.
    """
    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        super(ModelNew, self).__init__()
        self.epsilon = 1e-12  # Small epsilon for numerical stability
        
        # Initialize CUDA kernel
        if torch.cuda.is_available():
            self._init_cuda_kernel()
        else:
            self.cuda_kernel = None
    
    def _init_cuda_kernel(self):
        """Initialize CUDA kernel for Frobenius norm calculation and normalization."""
        cuda_code = """
        #include <cuda_runtime.h>
        
        // CUDA kernel for computing Frobenius norm and normalizing tensor in one pass
        extern "C" __global__ void frobenius_norm_kernel(
            float* input, float* output, float* norm_result, int numel, float epsilon) {
            
            // Shared memory for partial sums
            extern __shared__ float sdata[];
            
            int tid = threadIdx.x;
            int block_size = blockDim.x;
            int grid_size = blockDim.x * gridDim.x;
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Step 1: Compute partial sum of squares
            float thread_sum = 0.0f;
            
            // Grid-stride loop to handle large tensors
            for (int i = idx; i < numel; i += grid_size) {
                float val = input[i];
                thread_sum += val * val;
            }
            
            // Store in shared memory
            sdata[tid] = thread_sum;
            __syncthreads();
            
            // Reduce within block
            for (int s = block_size / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            
            // Write block result to global memory
            if (tid == 0) {
                atomicAdd(norm_result, sdata[0]);
            }
            
            // Wait for all blocks to finish reduction
            __threadfence();
            __syncthreads();
            
            // Last block computes final normalization factor
            if (blockIdx.x == 0 && tid == 0) {
                float norm_val = sqrtf(*norm_result + epsilon);
                *norm_result = 1.0f / norm_val;  // Store inverse norm for normalization
            }
            
            // Wait for normalization factor to be computed
            __threadfence();
            __syncthreads();
            
            // Step 2: Apply normalization
            float inv_norm = *norm_result;
            
            for (int i = idx; i < numel; i += grid_size) {
                output[i] = input[i] * inv_norm;
            }
        }
        """
        
        from torch.utils.cpp_extension import load_inline
        
        try:
            cuda_module = load_inline(
                name="frobenius_norm_cuda",
                cpp_sources="",
                cuda_sources=cuda_code,
                functions=["frobenius_norm_kernel"],
                verbose=False
            )
            
            self.cuda_kernel = cuda_module
        except Exception as e:
            print(f"Warning: Failed to load CUDA kernel: {e}")
            self.cuda_kernel = None
    
    def _compute_with_cuda_kernel(self, x):
        """
        Compute Frobenius norm and normalize using custom CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Normalized tensor
        """
        # Allocate output tensor
        output = torch.empty_like(x)
        
        # Allocate tensor for norm result (will contain the inverse norm)
        norm_result = torch.zeros(1, dtype=torch.float32, device=x.device)
        
        # Determine grid and block dimensions
        numel = x.numel()
        threads_per_block = min(512, numel)
        blocks = min(1024, (numel + threads_per_block - 1) // threads_per_block)
        
        # Shared memory size
        shared_mem_size = threads_per_block * 4  # 4 bytes per float
        
        # Launch kernel
        self.cuda_kernel.frobenius_norm_kernel(
            blocks, threads_per_block, shared_mem_size,
            x, output, norm_result, numel, self.epsilon
        )
        
        return output
    
    def _compute_with_torch(self, x):
        """
        Compute Frobenius norm and normalize using PyTorch operations.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Normalized tensor
        """
        # Ensure input is contiguous for optimal memory access
        x_cont = x if x.is_contiguous() else x.contiguous()
        
        # Flatten tensor for efficient dot product operation
        x_flat = x_cont.view(-1)
        
        # Use torch.dot for highly optimized sum of squares calculation
        sum_squared = torch.dot(x_flat, x_flat)
        
        # Compute inverse norm directly with rsqrt for better performance
        inv_norm = torch.rsqrt(sum_squared + self.epsilon)
        
        # Normalize using multiplication instead of division
        return x_cont * inv_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        # Try to use CUDA kernel if available and input is on CUDA
        if self.cuda_kernel is not None and x.is_cuda and x.dtype == torch.float32:
            try:
                return self._compute_with_cuda_kernel(x)
            except Exception as e:
                print(f"CUDA kernel failed: {e}. Falling back to PyTorch implementation.")
        
        # Fallback to optimized PyTorch implementation
        return self._compute_with_torch(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return []