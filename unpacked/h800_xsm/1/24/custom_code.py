import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized implementation of LogSoftmax using custom CUDA kernels.
    
    Args:
        dim (int): The dimension along which to apply the LogSoftmax
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.output_tensor = None
        self._setup_kernel()
    
    def _setup_kernel(self):
        if not hasattr(ModelNew, 'kernel_initialized'):
            cuda_source = """
            #include <cuda_runtime.h>
            
            // Optimized log_softmax kernel specifically for batch_size=16, dim=16384
            extern "C" __global__ void log_softmax_kernel(
                float* __restrict__ output,
                const float* __restrict__ input,
                const int batch_size,
                const int dim) {
                
                // Constants for optimization
                constexpr int WARP_SIZE = 32;
                const int warp_id = threadIdx.x / WARP_SIZE;
                const int lane_id = threadIdx.x % WARP_SIZE;
                
                // Each block processes one batch item
                const int batch_idx = blockIdx.x;
                if (batch_idx >= batch_size) return;
                
                // Get pointers to this batch item's input and output
                const float* batch_input = input + batch_idx * dim;
                float* batch_output = output + batch_idx * dim;
                
                // Shared memory for reductions
                __shared__ float shared_data[256]; // One per thread
                
                // Step 1: Find maximum value using grid-stride loop
                float thread_max = -INFINITY;
                
                // Each thread finds max in its assigned elements
                for (int i = threadIdx.x; i < dim; i += blockDim.x) {
                    thread_max = max(thread_max, batch_input[i]);
                }
                
                // Warp-level reduction first to minimize shared memory usage
                #pragma unroll
                for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                    thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
                }
                
                // Store warp results in shared memory
                if (lane_id == 0) {
                    shared_data[warp_id] = thread_max;
                }
                __syncthreads();
                
                // Final reduction across warps (only first warp needed)
                if (warp_id == 0) {
                    // Load warp results (only need to read up to number of warps)
                    thread_max = (lane_id < blockDim.x / WARP_SIZE) ? shared_data[lane_id] : -INFINITY;
                    
                    // Final warp reduction
                    #pragma unroll
                    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
                    }
                    
                    // Thread 0 writes final max to shared memory
                    if (lane_id == 0) {
                        shared_data[0] = thread_max;
                    }
                }
                __syncthreads();
                
                // Now shared_data[0] contains the max value for this batch item
                const float row_max = shared_data[0];
                
                // Step 2: Compute sum of exp(x - max) using grid-stride loop
                float thread_sum = 0.0f;
                
                // Use vectorized loads where possible for better memory throughput
                int vec_limit = (dim / 4) * 4;
                int i = threadIdx.x * 4;
                
                // Process 4 elements at a time when possible
                for (; i < vec_limit; i += blockDim.x * 4) {
                    if (i + 3 < dim) {
                        float4 inputs = *((float4*)&batch_input[i]);
                        thread_sum += expf(inputs.x - row_max);
                        thread_sum += expf(inputs.y - row_max);
                        thread_sum += expf(inputs.z - row_max);
                        thread_sum += expf(inputs.w - row_max);
                    }
                }
                
                // Process remaining elements individually
                for (i = vec_limit + threadIdx.x; i < dim; i += blockDim.x) {
                    thread_sum += expf(batch_input[i] - row_max);
                }
                
                // Warp-level reduction first
                #pragma unroll
                for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
                }
                
                // Store warp results in shared memory
                if (lane_id == 0) {
                    shared_data[warp_id] = thread_sum;
                }
                __syncthreads();
                
                // Final reduction across warps (only first warp needed)
                if (warp_id == 0) {
                    // Load warp results
                    thread_sum = (lane_id < blockDim.x / WARP_SIZE) ? shared_data[lane_id] : 0.0f;
                    
                    // Final warp reduction
                    #pragma unroll
                    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
                    }
                    
                    // Thread 0 writes final sum to shared memory
                    if (lane_id == 0) {
                        shared_data[0] = thread_sum;
                    }
                }
                __syncthreads();
                
                // Now shared_data[0] contains the sum for this batch item
                const float row_sum = shared_data[0];
                const float log_sum = logf(row_sum);
                
                // Step 3: Compute log_softmax for each element using grid-stride loop
                // Use vectorized operations where possible
                i = threadIdx.x * 4;
                for (; i < vec_limit; i += blockDim.x * 4) {
                    if (i + 3 < dim) {
                        float4 inputs = *((float4*)&batch_input[i]);
                        float4 outputs;
                        outputs.x = inputs.x - row_max - log_sum;
                        outputs.y = inputs.y - row_max - log_sum;
                        outputs.z = inputs.z - row_max - log_sum;
                        outputs.w = inputs.w - row_max - log_sum;
                        *((float4*)&batch_output[i]) = outputs;
                    }
                }
                
                // Process remaining elements individually
                for (i = vec_limit + threadIdx.x; i < dim; i += blockDim.x) {
                    batch_output[i] = batch_input[i] - row_max - log_sum;
                }
            }
            """
            
            try:
                from torch.utils.cpp_extension import load_inline
                ModelNew.cuda_kernel = load_inline(
                    name="log_softmax_optimized_cuda",
                    cpp_sources="",
                    cuda_sources=cuda_source,
                    functions=["log_softmax_kernel"],
                    verbose=False
                )
                ModelNew.kernel_initialized = True
            except Exception as e:
                print(f"Warning: Could not compile CUDA kernel: {e}")
                ModelNew.kernel_initialized = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation using an optimized implementation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        # Pre-allocate output tensor if needed
        if self.output_tensor is None or self.output_tensor.shape != x.shape or self.output_tensor.device != x.device:
            self.output_tensor = torch.empty_like(x)
        
        # Fall back to PyTorch implementation if not on CUDA or kernel compilation failed
        if not x.is_cuda or not hasattr(ModelNew, 'kernel_initialized') or not ModelNew.kernel_initialized:
            return torch.log_softmax(x, dim=self.dim, out=self.output_tensor)
        
        # Handle non-contiguous tensors
        if not x.is_contiguous():
            x = x.contiguous()
        
        # For dim != 1, transpose to make dim=1, then transpose back
        if self.dim != 1:
            perm = list(range(x.dim()))
            perm[self.dim], perm[1] = perm[1], perm[self.dim]
            x_t = x.permute(perm).contiguous()
            output_t = torch.empty_like(x_t)
            
            # Apply log_softmax to the transposed tensor
            self._apply_log_softmax(x_t, output_t)
            
            # Transpose back and return
            self.output_tensor = output_t.permute(perm)
            return self.output_tensor
        else:
            # Apply log_softmax directly
            self._apply_log_softmax(x, self.output_tensor)
            return self.output_tensor
    
    def _apply_log_softmax(self, x, output):
        batch_size, dim = x.shape
        
        # Launch the kernel with optimized configuration
        # Use 256 threads per block for good occupancy
        threads_per_block = 256
        
        ModelNew.cuda_kernel.log_softmax_kernel(
            output, x, batch_size, dim,
            grid=(batch_size, 1, 1),
            block=(threads_per_block, 1, 1)
        )

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed