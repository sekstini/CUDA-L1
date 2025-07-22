import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

class ModelNew(nn.Module):
    """
    Optimized implementation of Group Normalization using custom CUDA kernels.
    
    Args:
        num_features (int): Number of features in the input tensor.
        num_groups (int): Number of groups to divide the channels into.
    """
    def __init__(self, num_features, num_groups):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5
        
        # Validate that num_features is divisible by num_groups
        if num_features % num_groups != 0:
            raise ValueError('num_features must be divisible by num_groups')
        
        # Define the CUDA kernel code
        cuda_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <vector>

        // Warp-level reduction using shuffle instructions
        __device__ __forceinline__ float warpReduceSum(float val) {
            for (int offset = 16; offset > 0; offset /= 2)
                val += __shfl_down_sync(0xffffffff, val, offset);
            return val;
        }

        // Block-level reduction using warp shuffle and shared memory
        __device__ __forceinline__ float blockReduceSum(float val, float* shared) {
            const int tid = threadIdx.x;
            const int lane = tid & 31;
            const int wid = tid >> 5;
            const int warps_per_block = blockDim.x >> 5;
            
            // Warp reduction
            val = warpReduceSum(val);
            
            // Write reduced value to shared memory
            if (lane == 0)
                shared[wid] = val;
            
            __syncthreads();
            
            // Read from shared memory only if that warp exists
            val = (tid < warps_per_block) ? shared[lane] : 0.0f;
            
            // Final reduction within the first warp
            if (wid == 0)
                val = warpReduceSum(val);
            
            return val;
        }

        __global__ void group_norm_kernel(
            const float* __restrict__ input,
            float* __restrict__ output,
            const float* __restrict__ weight,
            const float* __restrict__ bias,
            const int batch_size,
            const int num_channels,
            const int height,
            const int width,
            const int num_groups,
            const int channels_per_group,
            const float eps) {
            
            // Each block processes one group in one batch element
            const int batch_idx = blockIdx.x / num_groups;
            const int group_idx = blockIdx.x % num_groups;
            
            if (batch_idx >= batch_size) return;
            
            // Calculate dimensions
            const int spatial_size = height * width;
            const int group_size = channels_per_group * spatial_size;
            
            // Shared memory for reduction and caching
            extern __shared__ float shared_mem[];
            float* s_sum = shared_mem;
            float* s_sq_sum = &shared_mem[blockDim.x / 32 + 1]; // +1 for padding
            float* s_weight = &shared_mem[2 * (blockDim.x / 32 + 1)];
            float* s_bias = &s_weight[channels_per_group];
            
            // Cache weights and biases in shared memory
            const int tid = threadIdx.x;
            if (tid < channels_per_group) {
                const int channel_idx = group_idx * channels_per_group + tid;
                s_weight[tid] = weight[channel_idx];
                s_bias[tid] = bias[channel_idx];
            }
            
            __syncthreads();
            
            // Thread local accumulators
            float thread_sum = 0.0f;
            float thread_sq_sum = 0.0f;
            
            // Calculate group offset in the input tensor
            const int group_offset = batch_idx * num_channels * spatial_size + 
                                    group_idx * channels_per_group * spatial_size;
            
            // Each thread processes multiple elements with stride
            const int stride = blockDim.x;
            
            // First pass: compute sum and sum of squares
            // Process by channel for better cache locality
            for (int c = 0; c < channels_per_group; c++) {
                const int channel_offset = c * spatial_size;
                
                // Process spatial elements with stride and loop unrolling
                for (int i = tid; i < spatial_size; i += stride * 4) {
                    // Process 4 elements at once when possible
                    float val1 = 0.0f, val2 = 0.0f, val3 = 0.0f, val4 = 0.0f;
                    
                    if (i < spatial_size)
                        val1 = input[group_offset + channel_offset + i];
                    if (i + stride < spatial_size)
                        val2 = input[group_offset + channel_offset + i + stride];
                    if (i + 2 * stride < spatial_size)
                        val3 = input[group_offset + channel_offset + i + 2 * stride];
                    if (i + 3 * stride < spatial_size)
                        val4 = input[group_offset + channel_offset + i + 3 * stride];
                    
                    // Accumulate sum and sum of squares
                    thread_sum += val1 + val2 + val3 + val4;
                    thread_sq_sum += val1 * val1 + val2 * val2 + val3 * val3 + val4 * val4;
                }
            }
            
            // Block-level reduction
            thread_sum = blockReduceSum(thread_sum, s_sum);
            thread_sq_sum = blockReduceSum(thread_sq_sum, s_sq_sum);
            
            // Compute mean and variance
            float mean, inv_std;
            if (tid == 0) {
                mean = thread_sum / group_size;
                float variance = fmaxf((thread_sq_sum / group_size) - (mean * mean), 0.0f);
                inv_std = rsqrtf(variance + eps);
                
                // Store for all threads to use
                s_sum[0] = mean;
                s_sq_sum[0] = inv_std;
            }
            
            __syncthreads();
            
            // Get the mean and inverse std
            mean = s_sum[0];
            inv_std = s_sq_sum[0];
            
            // Second pass: apply normalization, scale, and bias
            // Process by channel for better cache locality
            for (int c = 0; c < channels_per_group; c++) {
                const float w = s_weight[c];
                const float b = s_bias[c];
                const int channel_offset = c * spatial_size;
                
                // Process spatial elements with stride and loop unrolling
                for (int i = tid; i < spatial_size; i += stride * 4) {
                    // Process 4 elements at once when possible
                    if (i < spatial_size) {
                        const int idx = group_offset + channel_offset + i;
                        const float val = input[idx];
                        output[idx] = ((val - mean) * inv_std) * w + b;
                    }
                    
                    if (i + stride < spatial_size) {
                        const int idx = group_offset + channel_offset + i + stride;
                        const float val = input[idx];
                        output[idx] = ((val - mean) * inv_std) * w + b;
                    }
                    
                    if (i + 2 * stride < spatial_size) {
                        const int idx = group_offset + channel_offset + i + 2 * stride;
                        const float val = input[idx];
                        output[idx] = ((val - mean) * inv_std) * w + b;
                    }
                    
                    if (i + 3 * stride < spatial_size) {
                        const int idx = group_offset + channel_offset + i + 3 * stride;
                        const float val = input[idx];
                        output[idx] = ((val - mean) * inv_std) * w + b;
                    }
                }
            }
        }

        // C++ interface
        torch::Tensor group_norm_cuda_forward(
            const torch::Tensor& input,
            const torch::Tensor& weight,
            const torch::Tensor& bias,
            int num_groups,
            float eps) {
            
            // Get dimensions
            const auto batch_size = input.size(0);
            const auto num_channels = input.size(1);
            const auto height = input.size(2);
            const auto width = input.size(3);
            const int channels_per_group = num_channels / num_groups;
            
            // Create output tensor
            auto output = torch::empty_like(input);
            
            // Calculate launch parameters
            const int threads_per_block = 256; // Using 256 threads per block for good occupancy
            const int blocks = batch_size * num_groups;
            
            // Calculate shared memory size
            int shared_mem_size = 2 * (threads_per_block / 32 + 1) * sizeof(float); // For reduction
            shared_mem_size += 2 * channels_per_group * sizeof(float); // For weights and biases
            
            // Launch kernel
            AT_DISPATCH_FLOATING_TYPES(input.type(), "group_norm_cuda_forward", ([&] {
                if (std::is_same<scalar_t, float>::value) {
                    group_norm_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
                        input.data_ptr<float>(),
                        output.data_ptr<float>(),
                        weight.data_ptr<float>(),
                        bias.data_ptr<float>(),
                        batch_size,
                        num_channels,
                        height,
                        width,
                        num_groups,
                        channels_per_group,
                        eps
                    );
                } else {
                    // For non-float types, fall back to PyTorch implementation
                    output = torch::group_norm(input, num_groups, weight, bias, eps);
                }
            }));
            
            return output;
        }
        """

        cpp_source = """
        #include <torch/extension.h>

        // Forward declaration of CUDA functions
        torch::Tensor group_norm_cuda_forward(
            const torch::Tensor& input,
            const torch::Tensor& weight,
            const torch::Tensor& bias,
            int num_groups,
            float eps);

        // C++ interface
        torch::Tensor group_norm_forward(
            const torch::Tensor& input,
            const torch::Tensor& weight,
            const torch::Tensor& bias,
            int num_groups,
            float eps) {
            
            return group_norm_cuda_forward(input, weight, bias, num_groups, eps);
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("forward", &group_norm_forward, "GroupNorm forward (CUDA)");
        }
        """

        # Compile the extension on-the-fly
        try:
            self.groupnorm_cuda = load_inline(
                name="groupnorm_cuda_optimized",
                cpp_sources=cpp_source,
                cuda_sources=cuda_source,
                functions=["forward"],
                with_cuda=True,
                build_directory=os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions")
            )
        except Exception as e:
            print(f"Failed to compile CUDA extension: {e}")
            self.groupnorm_cuda = None
        
    def forward(self, x):
        """
        Applies Group Normalization to the input tensor using optimized CUDA kernel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).
            
        Returns:
            torch.Tensor: Output tensor with Group Normalization applied.
        """
        # Fall back to PyTorch implementation if CUDA is not available or compilation failed
        if not x.is_cuda or self.groupnorm_cuda is None:
            return torch.nn.functional.group_norm(
                x, self.num_groups, self.weight, self.bias, self.eps
            )
        
        # Use our optimized CUDA kernel implementation
        return self.groupnorm_cuda.forward(
            x, self.weight, self.bias, self.num_groups, self.eps
        )

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
features = 64
num_groups = 8
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features, num_groups]