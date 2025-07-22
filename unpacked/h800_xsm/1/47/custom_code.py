import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Define the CUDA kernel for sum reduction along dimension 1
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sum_reduction_dim1_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2) {
    
    // Each block handles one (batch_idx, dim2_idx) pair
    const int batch_idx = blockIdx.x;
    const int dim2_idx = blockIdx.y;
    
    // Early exit if out of bounds
    if (batch_idx >= batch_size || dim2_idx >= dim2) return;
    
    // Calculate input and output base indices
    const int input_base = batch_idx * dim1 * dim2 + dim2_idx;
    const int output_idx = batch_idx * dim2 + dim2_idx;
    
    // Each thread accumulates values across dim1
    scalar_t thread_sum = 0;
    
    // Each thread processes multiple elements with stride for better memory coalescing
    // For dim1=256, we want each thread to process multiple elements
    for (int i = threadIdx.x; i < dim1; i += blockDim.x) {
        thread_sum += input[input_base + i * dim2];
    }
    
    // Use shared memory for reduction
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    // Store thread sum in shared memory
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Final warp reduction (no __syncthreads needed within a warp)
    if (threadIdx.x < 32) {
        // Use warp shuffle operations for the final reduction steps
        scalar_t sum = shared_data[threadIdx.x];
        
        // Unroll the final warp reduction steps
        if (blockDim.x >= 64) sum += shared_data[threadIdx.x + 32];
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // Write result to output
        if (threadIdx.x == 0) {
            output[output_idx] = sum;
        }
    }
}

// Specialized kernel for float type with vectorized memory access
__global__ void sum_reduction_dim1_float_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2) {
    
    // Each block handles one (batch_idx, dim2_idx) pair
    const int batch_idx = blockIdx.x;
    const int dim2_idx = blockIdx.y;
    
    // Early exit if out of bounds
    if (batch_idx >= batch_size || dim2_idx >= dim2) return;
    
    // Calculate input and output base indices
    const int input_base = batch_idx * dim1 * dim2 + dim2_idx;
    const int output_idx = batch_idx * dim2 + dim2_idx;
    
    // Each thread accumulates values across dim1
    float thread_sum = 0.0f;
    
    // Process multiple elements per thread for better arithmetic intensity
    const int items_per_thread = 4;
    const int thread_stride = blockDim.x * items_per_thread;
    
    // Main loop - process elements in chunks
    for (int base_idx = threadIdx.x; base_idx < dim1; base_idx += thread_stride) {
        // Process up to items_per_thread elements if available
        #pragma unroll
        for (int offset = 0; offset < items_per_thread && base_idx + offset * blockDim.x < dim1; offset++) {
            int i = base_idx + offset * blockDim.x;
            thread_sum += input[input_base + i * dim2];
        }
    }
    
    // Use shared memory for reduction
    extern __shared__ char shared_mem[];
    float* shared_data = reinterpret_cast<float*>(shared_mem);
    
    // Store thread sum in shared memory
    shared_data[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Final warp reduction (no __syncthreads needed within a warp)
    if (threadIdx.x < 32) {
        // Use warp shuffle operations for the final reduction steps
        float sum = shared_data[threadIdx.x];
        
        // Unroll the final warp reduction steps
        if (blockDim.x >= 64) sum += shared_data[threadIdx.x + 32];
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // Write result to output
        if (threadIdx.x == 0) {
            output[output_idx] = sum;
        }
    }
}

torch::Tensor sum_reduction_dim1_cuda(torch::Tensor input) {
    const int batch_size = input.size(0);
    const int dim1 = input.size(1);
    const int dim2 = input.size(2);
    
    // Create output tensor
    auto output = torch::empty({batch_size, 1, dim2}, input.options());
    auto output_view = output.view({batch_size, dim2});
    
    // Configure kernel parameters
    const int threads_per_block = 256;
    const dim3 blocks(batch_size, dim2);
    const size_t shared_mem_size = threads_per_block * sizeof(float);
    
    // Launch kernel with specialized implementation for float
    if (input.scalar_type() == torch::kFloat) {
        sum_reduction_dim1_float_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
            input.data_ptr<float>(),
            output_view.data_ptr<float>(),
            batch_size,
            dim1,
            dim2
        );
    } else {
        // Generic implementation for other floating point types
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduction_dim1_kernel", ([&] {
            sum_reduction_dim1_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                output_view.data_ptr<scalar_t>(),
                batch_size,
                dim1,
                dim2
            );
        }));
    }
    
    return output;
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor sum_reduction_dim1(torch::Tensor input) {
    CHECK_INPUT(input);
    return sum_reduction_dim1_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_reduction_dim1", &sum_reduction_dim1, "Sum reduction along dimension 1");
}
"""

# Compile the CUDA extension
try:
    sum_reduction_cuda = load_inline(
        name="sum_reduction_cuda_optimized",
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=["sum_reduction_dim1"],
        verbose=False,
        with_cuda=True,
        extra_cuda_cflags=["-O3", "--use_fast_math"]
    )
except Exception as e:
    print(f"CUDA compilation failed: {e}")
    sum_reduction_cuda = None

class ModelNew(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.use_cuda_kernel = sum_reduction_cuda is not None and torch.cuda.is_available()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sum reduction over the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor after sum reduction, shape (..., 1, ...).
        """
        if self.use_cuda_kernel and self.dim == 1 and x.dim() == 3 and x.is_cuda:
            if not x.is_contiguous():
                x = x.contiguous()
            return sum_reduction_cuda.sum_reduction_dim1(x)
        else:
            return torch.sum(x, dim=self.dim, keepdim=True)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim1 = 256
dim2 = 256
reduce_dim = 1

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [reduce_dim]