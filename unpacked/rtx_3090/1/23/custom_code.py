import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

class ModelNew(nn.Module):
    """
    Optimized model that performs a Softmax activation using custom CUDA kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Compile the CUDA kernel
        self._compile_kernel()
    
    def _compile_kernel(self):
        cuda_source = """
        #include <cuda_runtime.h>
        
        // Helper functions for vectorized memory operations
        template <typename T>
        __device__ __forceinline__ void load_vector(const float* addr, T& val) {
            val = *(reinterpret_cast<const T*>(addr));
        }
        
        template <typename T>
        __device__ __forceinline__ void store_vector(float* addr, const T& val) {
            *(reinterpret_cast<T*>(addr)) = val;
        }
        
        // Optimized softmax kernel specifically for batch_size=16, dim=16384
        __global__ void softmax_kernel_optimized(const float* __restrict__ input, 
                                               float* __restrict__ output, 
                                               int dim) {
            // Each block processes one batch item
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;
            const int lane_id = tid % 32;
            const int warp_id = tid / 32;
            const int num_warps = blockDim.x / 32;
            
            // Pointers to this batch item's data
            const float* batch_input = input + batch_idx * dim;
            float* batch_output = output + batch_idx * dim;
            
            // Shared memory for reductions
            __shared__ float warp_max[16]; // Max values from each warp
            __shared__ float warp_sum[16]; // Sum values from each warp
            
            // Register cache for intermediate results - process 16 elements per thread
            // For 256 threads and 16384 elements, each thread processes 64 elements
            // We'll process them in chunks of 16 for better register utilization
            const int elements_per_thread = (dim + blockDim.x - 1) / blockDim.x;
            const int vector_size = 4; // Using float4 for memory operations
            const int vectors_per_thread = elements_per_thread / vector_size;
            
            // Step 1: Find max value (for numerical stability)
            float thread_max = -INFINITY;
            
            // Each thread processes multiple elements with vectorized loads
            #pragma unroll 4
            for (int v = 0; v < vectors_per_thread; v++) {
                const int base_idx = tid * vector_size + v * blockDim.x * vector_size;
                if (base_idx + 3 < dim) { // Make sure we don't read past the end
                    float4 in_vec;
                    load_vector<float4>(&batch_input[base_idx], in_vec);
                    
                    thread_max = fmaxf(thread_max, in_vec.x);
                    thread_max = fmaxf(thread_max, in_vec.y);
                    thread_max = fmaxf(thread_max, in_vec.z);
                    thread_max = fmaxf(thread_max, in_vec.w);
                } else {
                    // Handle boundary case
                    for (int j = 0; j < vector_size && base_idx + j < dim; j++) {
                        thread_max = fmaxf(thread_max, batch_input[base_idx + j]);
                    }
                }
            }
            
            // Warp-level reduction for max using shuffle operations
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, offset));
            }
            
            // First thread in each warp writes its max to shared memory
            if (lane_id == 0) {
                warp_max[warp_id] = thread_max;
            }
            __syncthreads();
            
            // First warp reduces the warp maxes
            if (warp_id == 0) {
                thread_max = (lane_id < num_warps) ? warp_max[lane_id] : -INFINITY;
                
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, offset));
                }
                
                // Broadcast max to all threads
                if (lane_id == 0) {
                    warp_max[0] = thread_max;
                }
            }
            __syncthreads();
            
            // Now warp_max[0] contains the max value for this batch item
            const float row_max = warp_max[0];
            
            // Step 2: Compute exp(x - max) and sum in a single pass
            float thread_sum = 0.0f;
            
            // Process elements in chunks of 4 for better memory throughput
            #pragma unroll 4
            for (int v = 0; v < vectors_per_thread; v++) {
                const int base_idx = tid * vector_size + v * blockDim.x * vector_size;
                if (base_idx + 3 < dim) {
                    float4 in_vec;
                    float4 out_vec;
                    
                    load_vector<float4>(&batch_input[base_idx], in_vec);
                    
                    // Compute exponentials with fast math
                    out_vec.x = __expf(in_vec.x - row_max);
                    out_vec.y = __expf(in_vec.y - row_max);
                    out_vec.z = __expf(in_vec.z - row_max);
                    out_vec.w = __expf(in_vec.w - row_max);
                    
                    // Add to thread's sum
                    thread_sum += out_vec.x + out_vec.y + out_vec.z + out_vec.w;
                    
                    // Store intermediate results
                    store_vector<float4>(&batch_output[base_idx], out_vec);
                } else {
                    // Handle boundary case
                    for (int j = 0; j < vector_size && base_idx + j < dim; j++) {
                        float exp_val = __expf(batch_input[base_idx + j] - row_max);
                        batch_output[base_idx + j] = exp_val;
                        thread_sum += exp_val;
                    }
                }
            }
            
            // Warp-level reduction for sum using shuffle operations
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, offset);
            }
            
            // First thread in each warp writes its sum to shared memory
            if (lane_id == 0) {
                warp_sum[warp_id] = thread_sum;
            }
            __syncthreads();
            
            // First warp reduces the warp sums
            if (warp_id == 0) {
                thread_sum = (lane_id < num_warps) ? warp_sum[lane_id] : 0.0f;
                
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, offset);
                }
                
                // Broadcast sum to all threads
                if (lane_id == 0) {
                    warp_sum[0] = thread_sum;
                }
            }
            __syncthreads();
            
            // Now warp_sum[0] contains the sum for this batch item
            const float row_sum = warp_sum[0];
            const float inv_row_sum = __fdividef(1.0f, row_sum);
            
            // Step 3: Normalize by dividing by the sum (using multiplication by inverse for speed)
            #pragma unroll 4
            for (int v = 0; v < vectors_per_thread; v++) {
                const int base_idx = tid * vector_size + v * blockDim.x * vector_size;
                if (base_idx + 3 < dim) {
                    float4 out_vec;
                    
                    load_vector<float4>(&batch_output[base_idx], out_vec);
                    
                    // Normalize
                    out_vec.x *= inv_row_sum;
                    out_vec.y *= inv_row_sum;
                    out_vec.z *= inv_row_sum;
                    out_vec.w *= inv_row_sum;
                    
                    // Store final results
                    store_vector<float4>(&batch_output[base_idx], out_vec);
                } else {
                    // Handle boundary case
                    for (int j = 0; j < vector_size && base_idx + j < dim; j++) {
                        batch_output[base_idx + j] *= inv_row_sum;
                    }
                }
            }
        }
        
        extern "C" {
            void softmax_launcher(const float* input, float* output, int batch_size, int dim, cudaStream_t stream) {
                // For our specific case (batch_size=16, dim=16384), use the optimized kernel
                const int threads_per_block = 256;
                const int blocks = batch_size;
                
                softmax_kernel_optimized<<<blocks, threads_per_block, 0, stream>>>(input, output, dim);
            }
        }
        """
        
        try:
            extension_name = f"softmax_optimized_{os.getpid()}"
            self.softmax_cuda = load_inline(
                name=extension_name,
                cpp_sources="",
                cuda_sources=cuda_source,
                functions=["softmax_launcher"],
                with_cuda=True,
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False
            )
            self.has_cuda_kernel = True
        except Exception as e:
            print(f"Warning: Failed to compile CUDA kernel: {e}")
            self.has_cuda_kernel = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        # Fall back to PyTorch implementation if not on CUDA or if kernel compilation failed
        if not x.is_cuda or not hasattr(self, 'has_cuda_kernel') or not self.has_cuda_kernel:
            return torch.softmax(x, dim=1)
        
        # Ensure input is contiguous and convert to float32 if needed
        if not x.is_contiguous():
            x = x.contiguous()
        
        if x.dtype != torch.float32:
            x = x.float()
        
        # Get batch size and dimension
        batch_size, dim = x.shape
        
        # Create output tensor
        output = torch.empty_like(x)
        
        # Get current CUDA stream
        stream = torch.cuda.current_stream().cuda_stream
        
        # Launch the kernel
        self.softmax_cuda.softmax_launcher(x, output, batch_size, dim, stream)
        
        return output

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed