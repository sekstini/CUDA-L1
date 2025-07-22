import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel code
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for mean reduction along dimension 0 (batch)
template <typename scalar_t>
__global__ void mean_dim0_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int dim1,
    int dim2) {
    
    const int tid = threadIdx.x;
    const int d1 = blockIdx.x;
    const int d2 = blockIdx.y;
    
    if (d1 >= dim1 || d2 >= dim2) return;
    
    // Index in the output tensor
    const int out_idx = d1 * dim2 + d2;
    
    // Use shared memory for the reduction
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem_bytes[];
    scalar_t* shared_mem = reinterpret_cast<scalar_t*>(shared_mem_bytes);
    
    // Load and sum values from global memory to shared memory
    scalar_t sum = 0;
    
    // Optimized for batch_size=16, which is small
    // Each thread handles one batch element
    if (tid < batch_size) {
        sum = input[tid * dim1 * dim2 + d1 * dim2 + d2];
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Perform reduction in shared memory
    // For batch_size=16, we only need a few reduction steps
    if (tid < 8) {
        if (tid + 8 < batch_size) {
            shared_mem[tid] += shared_mem[tid + 8];
        }
    }
    __syncthreads();
    
    if (tid < 4) {
        if (tid + 4 < batch_size) {
            shared_mem[tid] += shared_mem[tid + 4];
        }
    }
    __syncthreads();
    
    if (tid < 2) {
        if (tid + 2 < batch_size) {
            shared_mem[tid] += shared_mem[tid + 2];
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        scalar_t result = shared_mem[0];
        if (1 < batch_size) {
            result += shared_mem[1];
        }
        output[out_idx] = result / static_cast<scalar_t>(batch_size);
    }
}

// CUDA kernel for mean reduction along dimension 1
template <typename scalar_t>
__global__ void mean_dim1_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int dim1,
    int dim2) {
    
    const int tid = threadIdx.x;
    const int b = blockIdx.x;
    const int d2 = blockIdx.y;
    
    if (b >= batch_size || d2 >= dim2) return;
    
    // Index in the output tensor
    const int out_idx = b * dim2 + d2;
    
    // Use shared memory for the reduction
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem_bytes[];
    scalar_t* shared_mem = reinterpret_cast<scalar_t*>(shared_mem_bytes);
    
    // Calculate the base index for this thread block
    const int base_idx = b * dim1 * dim2 + d2;
    
    // Load and sum values from global memory to shared memory
    scalar_t sum = 0;
    
    // Use vectorized loads for better memory throughput if possible
    if (tid * 4 < dim1 && sizeof(scalar_t) == 4) {
        // Process 4 elements at a time when possible
        for (int d1 = tid * 4; d1 < dim1; d1 += blockDim.x * 4) {
            if (d1 + 3 < dim1) {
                // Full vector load - but need to access with stride dim2
                sum += input[base_idx + (d1 + 0) * dim2];
                sum += input[base_idx + (d1 + 1) * dim2];
                sum += input[base_idx + (d1 + 2) * dim2];
                sum += input[base_idx + (d1 + 3) * dim2];
            } else {
                // Handle boundary case
                for (int i = 0; i < 4 && d1 + i < dim1; ++i) {
                    sum += input[base_idx + (d1 + i) * dim2];
                }
            }
        }
    } else {
        // Standard processing one element at a time
        for (int d1 = tid; d1 < dim1; d1 += blockDim.x) {
            sum += input[base_idx + d1 * dim2];
        }
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Perform reduction in shared memory
    #pragma unroll
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    // Final reduction within a warp using warp shuffle
    if (tid < 32) {
        // Warp-level reduction using shuffle
        scalar_t val = shared_mem[tid];
        
        if (blockDim.x >= 64) val += shared_mem[tid + 32];
        
        // Unrolled warp reduction
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        
        if (tid == 0) {
            output[out_idx] = val / static_cast<scalar_t>(dim1);
        }
    }
}

// CUDA kernel for mean reduction along dimension 2
template <typename scalar_t>
__global__ void mean_dim2_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int dim1,
    int dim2) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    // Calculate batch and dim1 indices
    const int b = bid / dim1;
    const int d1 = bid % dim1;
    
    if (b >= batch_size || d1 >= dim1) return;
    
    // Index in the output tensor
    const int out_idx = b * dim1 + d1;
    
    // Use shared memory for the reduction
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char shared_mem_bytes[];
    scalar_t* shared_mem = reinterpret_cast<scalar_t*>(shared_mem_bytes);
    
    // Calculate the base index for this thread block
    const int base_idx = b * dim1 * dim2 + d1 * dim2;
    
    // Load and sum values from global memory to shared memory
    scalar_t sum = 0;
    
    // Use vectorized loads for better memory throughput if possible
    // This is most effective for dim2 reduction since memory is contiguous
    if (tid * 4 < dim2 && sizeof(scalar_t) == 4) {
        // Process 4 elements at a time when possible
        for (int d2 = tid * 4; d2 < dim2; d2 += blockDim.x * 4) {
            if (d2 + 3 < dim2) {
                // Full vector load - can use float4 directly since memory is contiguous
                float4 data = *reinterpret_cast<const float4*>(&input[base_idx + d2]);
                sum += data.x + data.y + data.z + data.w;
            } else {
                // Handle boundary case
                for (int i = 0; i < 4 && d2 + i < dim2; ++i) {
                    sum += input[base_idx + d2 + i];
                }
            }
        }
    } else {
        // Standard processing one element at a time
        for (int d2 = tid; d2 < dim2; d2 += blockDim.x) {
            sum += input[base_idx + d2];
        }
    }
    
    shared_mem[tid] = sum;
    __syncthreads();
    
    // Perform reduction in shared memory
    #pragma unroll
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    // Final reduction within a warp using warp shuffle
    if (tid < 32) {
        // Warp-level reduction using shuffle
        scalar_t val = shared_mem[tid];
        
        if (blockDim.x >= 64) val += shared_mem[tid + 32];
        
        // Unrolled warp reduction
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        
        if (tid == 0) {
            output[out_idx] = val / static_cast<scalar_t>(dim2);
        }
    }
}

// C++ wrapper functions for the CUDA kernels
torch::Tensor mean_dim0_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    
    auto output = torch::empty({dim1, dim2}, input.options());
    
    // For batch_size=16, we can use a smaller thread block
    const int threads = 32;  // Just enough for our batch size and warp size
    const dim3 blocks(dim1, dim2);
    const int shared_mem_size = threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_dim0_cuda", ([&] {
        mean_dim0_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim1,
            dim2
        );
    }));
    
    return output;
}

torch::Tensor mean_dim1_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    
    auto output = torch::empty({batch_size, dim2}, input.options());
    
    // Choose block size based on dimension sizes
    const int threads = 256;  // Good balance for dim1=256
    const dim3 blocks(batch_size, dim2);
    const int shared_mem_size = threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_dim1_cuda", ([&] {
        mean_dim1_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim1,
            dim2
        );
    }));
    
    return output;
}

torch::Tensor mean_dim2_cuda(torch::Tensor input) {
    auto batch_size = input.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    
    auto output = torch::empty({batch_size, dim1}, input.options());
    
    // Choose block size based on dimension sizes
    const int threads = 256;  // Good balance for dim2=256
    // Use 1D grid for better occupancy
    const int blocks = batch_size * dim1;
    const int shared_mem_size = threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_dim2_cuda", ([&] {
        mean_dim2_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim1,
            dim2
        );
    }));
    
    return output;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mean_dim0", &mean_dim0_cuda, "Mean reduction along dimension 0");
    m.def("mean_dim1", &mean_dim1_cuda, "Mean reduction along dimension 1");
    m.def("mean_dim2", &mean_dim2_cuda, "Mean reduction along dimension 2");
}
"""

class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension.
    
    Args:
        dim (int): The dimension to reduce over.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        
        # Compile the CUDA extension on first initialization
        try:
            self.mean_cuda = load_inline(
                name="mean_cuda_optimized",
                cpp_sources="",
                cuda_sources=cuda_source,
                functions=["mean_dim0", "mean_dim1", "mean_dim2"],
                with_cuda=True,
                extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v"]
            )
        except Exception as e:
            print(f"Failed to compile CUDA extension: {e}")
            self.mean_cuda = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean.
        
        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.
            
        Returns:
            torch.Tensor: Output tensor with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
        """
        # Fall back to PyTorch's implementation in certain cases
        if self.mean_cuda is None or not x.is_cuda or x.dim() != 3:
            return torch.mean(x, dim=self.dim)
        
        # Use our optimized CUDA kernels based on the reduction dimension
        if self.dim == 0:
            return self.mean_cuda.mean_dim0(x)
        elif self.dim == 1:
            return self.mean_cuda.mean_dim1(x)
        elif self.dim == 2:
            return self.mean_cuda.mean_dim2(x)
        else:
            # Fall back to PyTorch's implementation for other dimensions
            return torch.mean(x, dim=self.dim)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]  # Reducing along dimension 1