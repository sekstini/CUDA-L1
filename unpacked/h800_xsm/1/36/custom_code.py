import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    """
    Optimized implementation of RMS Normalization.
    
    Args:
        num_features (int): Number of features in the input tensor.
        eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        
        # Try to load custom CUDA kernel
        self.use_custom_kernel = torch.cuda.is_available()
        if self.use_custom_kernel:
            try:
                # Define the optimized CUDA kernel
                cuda_kernel = """
                #include <cuda_runtime.h>
                
                extern "C" __global__ void rmsnorm_kernel(
                    float* __restrict__ output,
                    const float* __restrict__ input,
                    const int batch_size,
                    const int features,
                    const int dim1,
                    const int dim2,
                    const float eps) {
                    
                    // Get batch index
                    const int b = blockIdx.x;
                    if (b >= batch_size) return;
                    
                    // Calculate total spatial elements
                    const int total_spatial = dim1 * dim2;
                    
                    // Calculate offsets
                    const int elements_per_batch = features * total_spatial;
                    const int batch_offset = b * elements_per_batch;
                    
                    // Process multiple spatial positions per thread block
                    for (int spatial_idx = blockIdx.y * blockDim.y + threadIdx.y; 
                         spatial_idx < total_spatial; 
                         spatial_idx += gridDim.y * blockDim.y) {
                         
                        // Calculate spatial position
                        const int d1 = spatial_idx / dim2;
                        const int d2 = spatial_idx % dim2;
                        
                        // Thread identifiers for reduction
                        const int tid = threadIdx.x;
                        const int lane_id = tid % 32;
                        const int warp_id = tid / 32;
                        const int warps_per_block = (blockDim.x + 31) / 32;
                        
                        // Shared memory for warp sums
                        __shared__ float warp_sums[32];  // Support up to 32 warps
                        
                        // Calculate squared sum for this position across features
                        float sum_squared = 0.0f;
                        
                        // Each thread processes multiple features with loop unrolling
                        #pragma unroll 4
                        for (int f = tid; f < features; f += blockDim.x) {
                            const int idx = batch_offset + f * total_spatial + spatial_idx;
                            const float val = input[idx];
                            sum_squared += val * val;
                        }
                        
                        // Warp-level reduction using shuffle
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset /= 2) {
                            sum_squared += __shfl_down_sync(0xffffffff, sum_squared, offset);
                        }
                        
                        // First thread in each warp has the sum for its warp
                        if (lane_id == 0) {
                            warp_sums[warp_id] = sum_squared;
                        }
                        
                        __syncthreads();
                        
                        // First warp reduces all warp sums
                        if (warp_id == 0) {
                            float warp_sum = (lane_id < warps_per_block) ? warp_sums[lane_id] : 0.0f;
                            
                            // Reduce within the warp
                            #pragma unroll
                            for (int offset = 16; offset > 0; offset /= 2) {
                                warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
                            }
                            
                            // First thread has the final sum
                            if (lane_id == 0) {
                                warp_sums[0] = warp_sum;
                            }
                        }
                        
                        __syncthreads();
                        
                        // Get the final sum
                        const float final_sum = warp_sums[0];
                        
                        // Calculate RMS (root mean square)
                        const float mean_squared = final_sum / features;
                        const float inv_rms = rsqrtf(mean_squared + eps);
                        
                        // Normalize the input using the computed inv_rms
                        #pragma unroll 4
                        for (int f = tid; f < features; f += blockDim.x) {
                            const int idx = batch_offset + f * total_spatial + spatial_idx;
                            output[idx] = input[idx] * inv_rms;
                        }
                    }
                }
                """
                
                # Load the custom CUDA kernel
                from torch.utils.cpp_extension import load_inline
                self.rmsnorm_cuda = load_inline(
                    name="rmsnorm_cuda",
                    cpp_sources="",
                    cuda_sources=cuda_kernel,
                    functions=["rmsnorm_kernel"],
                    with_cuda=True,
                    verbose=False
                )
                self.custom_kernel_loaded = True
            except Exception as e:
                print(f"Failed to load custom CUDA kernel: {e}")
                self.custom_kernel_loaded = False
        else:
            self.custom_kernel_loaded = False
            
        # Pre-compute scaling factor for the fallback implementation
        self.register_buffer('inv_sqrt_features', torch.tensor(1.0 / math.sqrt(num_features)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization with maximum GPU optimization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        # Ensure optimal memory layout
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Use custom CUDA kernel if available and input is on CUDA
        if self.custom_kernel_loaded and x.is_cuda:
            output = torch.empty_like(x)
            batch_size, features, dim1, dim2 = x.shape
            total_spatial = dim1 * dim2
            
            # Configure thread block dimensions
            threads_x = min(256, features)  # Process features in parallel
            threads_y = min(4, total_spatial)  # Process spatial positions in parallel
            
            # Calculate grid dimensions
            blocks_x = batch_size
            blocks_y = min(1024, (total_spatial + threads_y - 1) // threads_y)
            
            # Launch the kernel
            self.rmsnorm_cuda.rmsnorm_kernel(
                grid=(blocks_x, blocks_y, 1),
                block=(threads_x, threads_y, 1),
                args=[output.data_ptr(), x.data_ptr(), batch_size, features, dim1, dim2, self.eps]
            )
            return output
            
        # Fallback to optimized PyTorch implementation
        # Use torch.linalg.vector_norm for more efficient norm computation
        norm = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
        
        # Scale by 1/sqrt(num_features) to get RMS value
        rms = norm * self.inv_sqrt_features
        
        # Add epsilon and compute reciprocal square root in one fused operation
        inv_rms = torch.rsqrt(rms.pow(2) + self.eps)
        
        # Final normalization with optimized multiplication
        return x * inv_rms

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]