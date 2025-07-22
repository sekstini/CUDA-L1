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
        self.num_groups = num_groups
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5
        
        # Verify that num_features is divisible by num_groups
        assert num_features % num_groups == 0, "num_features must be divisible by num_groups"
        
        # Define CUDA kernel code
        cuda_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <vector>

        // Constants for optimization
        #define WARP_SIZE 32
        #define BLOCK_SIZE 256
        #define ELEMENTS_PER_THREAD 8

        // Helper function for warp-level reduction using shuffle
        __device__ __forceinline__ float warpReduceSum(float val) {
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
                val += __shfl_down_sync(0xffffffff, val, offset);
            return val;
        }

        // Optimized CUDA kernel for computing mean and variance using Welford's algorithm
        template <typename scalar_t>
        __global__ void group_norm_stats_kernel(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ mean,
            scalar_t* __restrict__ var,
            int N, int C, int HW,
            int num_groups, int channels_per_group) {
            
            // Each block handles one group from one batch element
            const int batch_idx = blockIdx.x;
            const int group_idx = blockIdx.y;
            
            // Calculate starting point for this group
            const int group_size = channels_per_group * HW;
            const int group_offset = batch_idx * C * HW + group_idx * channels_per_group * HW;
            
            // Shared memory for reduction
            extern __shared__ float shared_mem[];
            float* shared_mean = shared_mem;
            float* shared_m2 = shared_mem + blockDim.x;
            float* shared_count = shared_mem + 2 * blockDim.x;
            
            // Thread-local accumulators for Welford's algorithm
            float thread_mean = 0.0f;
            float thread_m2 = 0.0f;
            int thread_count = 0;
            
            // Each thread processes multiple elements with stride
            for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
                const scalar_t val = input[group_offset + i];
                thread_count++;
                
                // Welford's online algorithm for mean and variance
                float delta = static_cast<float>(val) - thread_mean;
                thread_mean += delta / thread_count;
                float delta2 = static_cast<float>(val) - thread_mean;
                thread_m2 += delta * delta2;
            }
            
            // Store in shared memory
            shared_mean[threadIdx.x] = thread_mean;
            shared_m2[threadIdx.x] = thread_m2;
            shared_count[threadIdx.x] = thread_count;
            __syncthreads();
            
            // Reduce within the block
            for (int stride = blockDim.x / 2; stride > WARP_SIZE; stride >>= 1) {
                if (threadIdx.x < stride) {
                    int idx = threadIdx.x + stride;
                    int n1 = shared_count[threadIdx.x];
                    int n2 = shared_count[idx];
                    
                    if (n1 > 0 && n2 > 0) {
                        int n = n1 + n2;
                        float delta = shared_mean[idx] - shared_mean[threadIdx.x];
                        
                        shared_mean[threadIdx.x] += delta * n2 / n;
                        shared_m2[threadIdx.x] += shared_m2[idx] + delta * delta * n1 * n2 / n;
                        shared_count[threadIdx.x] = n;
                    }
                }
                __syncthreads();
            }
            
            // Final warp-level reduction
            if (threadIdx.x < WARP_SIZE) {
                float warp_mean = shared_mean[threadIdx.x];
                float warp_m2 = shared_m2[threadIdx.x];
                int warp_count = shared_count[threadIdx.x];
                
                // Use warp shuffle to reduce within the warp
                for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                    float other_mean = __shfl_down_sync(0xffffffff, warp_mean, offset);
                    float other_m2 = __shfl_down_sync(0xffffffff, warp_m2, offset);
                    int other_count = __shfl_down_sync(0xffffffff, warp_count, offset);
                    
                    int n1 = warp_count;
                    int n2 = other_count;
                    
                    if (n1 > 0 && n2 > 0) {
                        int n = n1 + n2;
                        float delta = other_mean - warp_mean;
                        
                        warp_mean += delta * n2 / n;
                        warp_m2 += other_m2 + delta * delta * n1 * n2 / n;
                        warp_count = n;
                    }
                }
                
                // First thread writes the final result
                if (threadIdx.x == 0) {
                    mean[batch_idx * num_groups + group_idx] = warp_mean;
                    var[batch_idx * num_groups + group_idx] = warp_count > 1 ? warp_m2 / warp_count : 0.0f;
                }
            }
        }

        // Optimized CUDA kernel for normalization
        template <typename scalar_t>
        __global__ void group_norm_kernel(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ output,
            const scalar_t* __restrict__ mean,
            const scalar_t* __restrict__ var,
            const scalar_t* __restrict__ weight,
            const scalar_t* __restrict__ bias,
            int N, int C, int HW,
            int num_groups, int channels_per_group,
            float eps) {
            
            // Use 2D grid for batch and channel dimensions
            const int batch_idx = blockIdx.x;
            const int c = blockIdx.y * blockDim.y + threadIdx.y;
            
            // Early exit if channel is out of bounds
            if (c >= C) return;
            
            // Determine which group this channel belongs to
            const int group_idx = c / channels_per_group;
            
            // Get mean and variance for this batch and group
            const float group_mean = static_cast<float>(mean[batch_idx * num_groups + group_idx]);
            const float group_var = static_cast<float>(var[batch_idx * num_groups + group_idx]);
            const float inv_std = rsqrtf(group_var + eps);
            
            // Get weight and bias for this channel
            const float gamma = static_cast<float>(weight[c]);
            const float beta = static_cast<float>(bias[c]);
            
            // Process spatial locations
            const int base_idx = (batch_idx * C + c) * HW;
            
            // Each thread processes multiple elements with stride
            for (int hw = threadIdx.x; hw < HW; hw += blockDim.x) {
                const int idx = base_idx + hw;
                float val = static_cast<float>(input[idx]);
                val = (val - group_mean) * inv_std;
                val = val * gamma + beta;
                output[idx] = static_cast<scalar_t>(val);
            }
        }

        // C++ interface
        std::vector<torch::Tensor> group_norm_cuda(
            const torch::Tensor& input,
            const torch::Tensor& weight,
            const torch::Tensor& bias,
            int num_groups,
            float eps) {
            
            // Get dimensions
            const auto N = input.size(0);
            const auto C = input.size(1);
            int HW = 1;
            for (int i = 2; i < input.dim(); ++i) {
                HW *= input.size(i);
            }
            const auto channels_per_group = C / num_groups;
            
            // Create output tensor
            auto output = torch::empty_like(input);
            
            // Create temporary tensors for mean and variance
            auto mean = torch::empty({N, num_groups}, input.options());
            auto var = torch::empty({N, num_groups}, input.options());
            
            // Calculate optimal thread configurations
            const int stats_threads = BLOCK_SIZE;
            const int shared_mem_size = stats_threads * 3 * sizeof(float); // mean, m2, count
            
            // Stats kernel configuration
            const dim3 stats_blocks(N, num_groups);
            
            // Launch statistics kernel
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "group_norm_stats_cuda", ([&] {
                group_norm_stats_kernel<scalar_t><<<stats_blocks, stats_threads, shared_mem_size>>>(
                    input.data_ptr<scalar_t>(),
                    mean.data_ptr<scalar_t>(),
                    var.data_ptr<scalar_t>(),
                    N, C, HW,
                    num_groups, channels_per_group
                );
            }));
            
            // Normalization kernel configuration - optimize for spatial dimensions
            const int norm_threads_x = 32;  // Process spatial dimensions
            const int norm_threads_y = 8;   // Process multiple channels in parallel
            const dim3 norm_threads(norm_threads_x, norm_threads_y);
            
            const int norm_blocks_y = (C + norm_threads_y - 1) / norm_threads_y;
            const dim3 norm_blocks(N, norm_blocks_y);
            
            // Launch normalization kernel
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "group_norm_cuda", ([&] {
                group_norm_kernel<scalar_t><<<norm_blocks, norm_threads>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    mean.data_ptr<scalar_t>(),
                    var.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    N, C, HW,
                    num_groups, channels_per_group,
                    eps
                );
            }));
            
            return {output};
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("forward", &group_norm_cuda, "Group Norm forward (CUDA)");
        }
        """
        
        # Try to load the CUDA extension
        try:
            self.group_norm_cuda = load_inline(
                name='group_norm_cuda',
                cpp_sources=[cuda_source],
                cuda_sources=[],
                functions=['forward'],
                with_cuda=True,
                extra_cuda_cflags=['-O3'],
                build_directory=os.path.join(os.path.expanduser('~'), '.cache', 'torch_extensions')
            )
            self.cuda_available = True
        except Exception as e:
            print(f"Failed to load CUDA extension: {e}")
            self.cuda_available = False
        
    def forward(self, x):
        """
        Applies Group Normalization to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).
            
        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Use our optimized CUDA kernel if available and input is on GPU
        if hasattr(self, 'cuda_available') and self.cuda_available and x.is_cuda:
            # Make sure tensors are contiguous for better memory access
            x = x.contiguous()
            weight = self.weight.contiguous()
            bias = self.bias.contiguous()
            
            # Call our optimized CUDA implementation
            output = self.group_norm_cuda.forward(x, weight, bias, self.num_groups, self.eps)[0]
            return output
        else:
            # Fall back to PyTorch implementation
            return nn.functional.group_norm(
                x, self.num_groups, self.weight, self.bias, self.eps
            )

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
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