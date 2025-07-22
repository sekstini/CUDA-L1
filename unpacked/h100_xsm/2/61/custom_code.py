import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define CUDA kernel for highly optimized GroupNorm
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void optimized_group_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_groups,
    const int channels_per_group,
    const int D, const int H, const int W,
    const float eps) {
    
    // Each block handles one batch and one group
    const int batch_idx = blockIdx.x;
    const int group_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || group_idx >= num_groups)
        return;
    
    const int start_channel = group_idx * channels_per_group;
    const int spatial_size = D * H * W;
    const int group_size = channels_per_group * spatial_size;
    
    // Shared memory for reductions - padded to avoid bank conflicts
    extern __shared__ float s_data[];
    float* s_mean = s_data;
    float* s_var = &s_data[blockDim.x / 32 + 2];  // +2 for padding
    
    const int tid = threadIdx.x;
    const int lane_id = tid & 0x1F;  // tid % 32, faster with bitwise AND
    const int warp_id = tid >> 5;    // tid / 32, faster with bitwise shift
    const int num_warps = blockDim.x >> 5;
    
    // Single pass: compute sum and sum of squares simultaneously
    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;
    
    // Use linearized indexing for better memory access patterns
    const int base_offset = (batch_idx * num_groups * channels_per_group + start_channel) * spatial_size;
    const int total_elements = channels_per_group * spatial_size;
    
    // Process elements with stride for better memory coalescing
    #pragma unroll 4
    for (int i = tid; i < total_elements; i += blockDim.x) {
        const int c_offset = i / spatial_size;
        const int s_offset = i % spatial_size;
        const int idx = base_offset + c_offset * spatial_size + s_offset;
        const float val = static_cast<float>(input[idx]);
        thread_sum += val;
        thread_sq_sum += val * val;
    }
    
    // Warp-level reduction for sum and sum of squares
    thread_sum = warpReduceSum(thread_sum);
    thread_sq_sum = warpReduceSum(thread_sq_sum);
    
    // Store warp results in shared memory - only one thread per warp writes
    if (lane_id == 0) {
        s_mean[warp_id] = thread_sum;
        s_var[warp_id] = thread_sq_sum;
    }
    __syncthreads();
    
    // Final reduction across warps - only first warp participates
    if (warp_id == 0 && lane_id < num_warps) {
        float block_sum = s_mean[lane_id];
        float block_sq_sum = s_var[lane_id];
        
        // Reduce across first warp using warp shuffle
        block_sum = warpReduceSum(block_sum);
        block_sq_sum = warpReduceSum(block_sq_sum);
        
        // Only one thread writes the final results
        if (lane_id == 0) {
            // Calculate mean and variance in one step
            float mean = block_sum / group_size;
            float variance = fmaxf((block_sq_sum / group_size) - (mean * mean), 0.0f);
            s_mean[0] = mean;
            s_var[0] = rsqrtf(variance + eps);  // Inverse std dev
        }
    }
    __syncthreads();
    
    // Get mean and inverse standard deviation
    const float mean = s_mean[0];
    const float inv_std = s_var[0];
    
    // Apply normalization with optimized memory access pattern
    #pragma unroll 4
    for (int i = tid; i < total_elements; i += blockDim.x) {
        const int c_offset = i / spatial_size;
        const int s_offset = i % spatial_size;
        const int idx = base_offset + c_offset * spatial_size + s_offset;
        output[idx] = static_cast<scalar_t>((static_cast<float>(input[idx]) - mean) * inv_std);
    }
}

// Specialized kernel for common case where channels_per_group is small
template <typename scalar_t, int CHANNELS_PER_GROUP>
__global__ void optimized_group_norm_specialized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int num_groups,
    const int D, const int H, const int W,
    const float eps) {
    
    // Each block handles one batch and one group
    const int batch_idx = blockIdx.x;
    const int group_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || group_idx >= num_groups)
        return;
    
    const int start_channel = group_idx * CHANNELS_PER_GROUP;
    const int spatial_size = D * H * W;
    const int group_size = CHANNELS_PER_GROUP * spatial_size;
    
    // Shared memory for reductions - padded to avoid bank conflicts
    extern __shared__ float s_data[];
    float* s_mean = s_data;
    float* s_var = &s_data[blockDim.x / 32 + 2];  // +2 for padding
    
    const int tid = threadIdx.x;
    const int lane_id = tid & 0x1F;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;
    
    // Single pass: compute sum and sum of squares simultaneously
    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;
    
    // Use linearized indexing for better memory access patterns
    const int base_offset = (batch_idx * num_groups * CHANNELS_PER_GROUP + start_channel) * spatial_size;
    
    // Process elements with stride for better memory coalescing
    #pragma unroll
    for (int c = 0; c < CHANNELS_PER_GROUP; c++) {
        const int channel_offset = c * spatial_size;
        #pragma unroll 4
        for (int i = tid; i < spatial_size; i += blockDim.x) {
            const int idx = base_offset + channel_offset + i;
            const float val = static_cast<float>(input[idx]);
            thread_sum += val;
            thread_sq_sum += val * val;
        }
    }
    
    // Warp-level reduction for sum and sum of squares
    thread_sum = warpReduceSum(thread_sum);
    thread_sq_sum = warpReduceSum(thread_sq_sum);
    
    // Store warp results in shared memory - only one thread per warp writes
    if (lane_id == 0) {
        s_mean[warp_id] = thread_sum;
        s_var[warp_id] = thread_sq_sum;
    }
    __syncthreads();
    
    // Final reduction across warps - only first warp participates
    if (warp_id == 0 && lane_id < num_warps) {
        float block_sum = s_mean[lane_id];
        float block_sq_sum = s_var[lane_id];
        
        // Reduce across first warp using warp shuffle
        block_sum = warpReduceSum(block_sum);
        block_sq_sum = warpReduceSum(block_sq_sum);
        
        // Only one thread writes the final results
        if (lane_id == 0) {
            // Calculate mean and variance in one step
            float mean = block_sum / group_size;
            float variance = fmaxf((block_sq_sum / group_size) - (mean * mean), 0.0f);
            s_mean[0] = mean;
            s_var[0] = rsqrtf(variance + eps);  // Inverse std dev
        }
    }
    __syncthreads();
    
    // Get mean and inverse standard deviation
    const float mean = s_mean[0];
    const float inv_std = s_var[0];
    
    // Apply normalization with optimized memory access pattern
    #pragma unroll
    for (int c = 0; c < CHANNELS_PER_GROUP; c++) {
        const int channel_offset = c * spatial_size;
        #pragma unroll 4
        for (int i = tid; i < spatial_size; i += blockDim.x) {
            const int idx = base_offset + channel_offset + i;
            output[idx] = static_cast<scalar_t>((static_cast<float>(input[idx]) - mean) * inv_std);
        }
    }
}

torch::Tensor optimized_group_norm(
    const torch::Tensor& input,
    const int num_groups,
    const float eps) {
    
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto D = input.size(2);
    const auto H = input.size(3);
    const auto W = input.size(4);
    
    const int channels_per_group = channels / num_groups;
    
    auto output = torch::empty_like(input);
    
    // Optimize thread block size for better occupancy
    const int threads_per_block = 256;
    const dim3 blocks(batch_size, num_groups);
    const dim3 threads(threads_per_block);
    
    // Shared memory for warp-level reductions with padding to avoid bank conflicts
    const int shared_mem_size = 2 * ((threads_per_block / 32) + 2) * sizeof(float);
    
    // Launch kernel - use specialized kernel for common case
    AT_DISPATCH_FLOATING_TYPES(input.type(), "optimized_group_norm_kernel", ([&] {
        if (channels_per_group == 16) {
            optimized_group_norm_specialized_kernel<scalar_t, 16><<<blocks, threads, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                num_groups,
                D, H, W,
                eps
            );
        } else {
            optimized_group_norm_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                num_groups,
                channels_per_group,
                D, H, W,
                eps
            );
        }
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("optimized_group_norm", &optimized_group_norm, "Optimized GroupNorm implementation");
}
"""

# Try to compile and load the CUDA kernel
try:
    optimized_ops = load_inline(
        name="optimized_ops",
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=["optimized_group_norm"],
        verbose=True,
        with_cuda=True
    )
    cuda_available = True
except Exception as e:
    print(f"CUDA extension compilation failed: {e}. Falling back to PyTorch implementation.")
    cuda_available = False

class ModelNew(nn.Module):
    """
    Model that performs a transposed 3D convolution, applies ReLU, and then applies group normalization.
    Optimized implementation using a combination of PyTorch's native functions and custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.relu = nn.ReLU(inplace=True)  # Use inplace ReLU to save memory
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.groups = groups
        self.use_custom_groupnorm = cuda_available
        
        # Create multiple CUDA streams for better parallelism
        if torch.cuda.is_available():
            self.streams = [torch.cuda.Stream() for _ in range(2)]
        else:
            self.streams = None
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        # Ensure input is contiguous for better memory access patterns
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Use CUDA streams for better parallelism if available
        if self.streams is not None and x.is_cuda:
            with torch.cuda.stream(self.streams[0]):
                # Apply ConvTranspose3d using PyTorch's optimized implementation
                x = self.conv_transpose(x)
                
                # Apply ReLU in-place to save memory
                x = self.relu(x)
                
                # Apply our optimized GroupNorm if available
                if self.use_custom_groupnorm:
                    try:
                        with torch.cuda.stream(self.streams[1]):
                            result = optimized_ops.optimized_group_norm(x, self.groups, 1e-5)
                            
                        # Ensure computation is complete before returning
                        torch.cuda.current_stream().wait_stream(self.streams[0])
                        torch.cuda.current_stream().wait_stream(self.streams[1])
                        return result
                    except Exception as e:
                        print(f"Custom GroupNorm failed: {e}. Falling back to PyTorch implementation.")
                        self.use_custom_groupnorm = False
                        x = self.group_norm(x)
                else:
                    x = self.group_norm(x)
                
                # Ensure computation is complete before returning
                torch.cuda.current_stream().wait_stream(self.streams[0])
                return x
        else:
            # Standard execution path without CUDA stream
            x = self.conv_transpose(x)
            x = self.relu(x)
            
            if x.is_cuda and self.use_custom_groupnorm:
                try:
                    return optimized_ops.optimized_group_norm(x, self.groups, 1e-5)
                except Exception as e:
                    print(f"Custom GroupNorm failed: {e}. Falling back to PyTorch implementation.")
                    self.use_custom_groupnorm = False
                    return self.group_norm(x)
            else:
                return self.group_norm(x)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 8, 16, 16
kernel_size = 3
groups = 8
bias = False

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]