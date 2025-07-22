import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128

template <typename scalar_t>
__global__ void cumprod_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int seq_len) {
    
    // Each thread block processes one batch item
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    // Get pointers to the current batch item's data
    const scalar_t* batch_input = input + batch_idx * seq_len;
    scalar_t* batch_output = output + batch_idx * seq_len;
    
    // Shared memory for partial products
    __shared__ scalar_t s_partial_products[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int elements_per_thread = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int start_idx = tid * elements_per_thread;
    int end_idx = min(start_idx + elements_per_thread, seq_len);
    
    // First element in each thread's segment
    if (start_idx < seq_len) {
        batch_output[start_idx] = batch_input[start_idx];
    }
    
    // Compute local cumulative product for each thread's segment
    for (int i = start_idx + 1; i < end_idx; i++) {
        batch_output[i] = batch_output[i-1] * batch_input[i];
    }
    
    // Store the final product of each thread's segment
    if (end_idx > start_idx && end_idx <= seq_len) {
        s_partial_products[tid] = batch_output[end_idx - 1];
    } else {
        s_partial_products[tid] = 1.0;
    }
    
    __syncthreads();
    
    // Compute prefix products for the partial products
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        scalar_t temp = 1.0;
        if (tid >= stride) {
            temp = s_partial_products[tid - stride];
        }
        __syncthreads();
        
        if (tid >= stride) {
            s_partial_products[tid] *= temp;
        }
        __syncthreads();
    }
    
    // Apply the prefix products to subsequent segments
    for (int t = 0; t < tid; t++) {
        int t_start = t * elements_per_thread;
        int t_end = min(t_start + elements_per_thread, seq_len);
        
        if (t_end > t_start && start_idx < seq_len) {
            scalar_t multiplier = s_partial_products[t];
            for (int i = start_idx; i < end_idx; i++) {
                batch_output[i] *= multiplier;
            }
        }
    }
}

torch::Tensor cumprod_cuda(torch::Tensor input, int dim) {
    // Only support dim=1 for 2D tensors
    TORCH_CHECK(input.dim() == 2 && dim == 1, 
                "Custom CUDA kernel only supports 2D tensors with dim=1");
    
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    
    // Create output tensor with same shape and dtype
    auto output = torch::empty_like(input);
    
    // Calculate grid and block dimensions
    int threads_per_block = BLOCK_SIZE;
    int blocks = batch_size;
    
    // Launch kernel with dynamic dispatch based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda", ([&] {
        cumprod_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            seq_len);
    }));
    
    return output;
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor cumprod_cuda(torch::Tensor input, int dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumprod", &cumprod_cuda, "Cumulative product (CUDA)");
}
"""

class ModelNew(nn.Module):
    """
    A model that performs a cumulative product operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cuda_module = None
        
        # Lazy initialization of CUDA module
        if torch.cuda.is_available():
            try:
                self.cuda_module = load_inline(
                    name="cumprod_cuda",
                    cpp_sources=[cpp_source],
                    cuda_sources=[cuda_source],
                    functions=["cumprod"],
                    verbose=False
                )
            except Exception as e:
                print(f"CUDA module compilation failed: {e}")
                self.cuda_module = None

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        # Use our custom CUDA kernel if available and applicable
        if (self.cuda_module is not None and x.is_cuda and x.dim() == 2 and 
            self.dim == 1 and x.dtype in [torch.float32, torch.float16, torch.float64]):
            return self.cuda_module.cumprod(x, self.dim)
        
        # Fall back to PyTorch implementation for other cases
        return torch.cumprod(x, dim=self.dim)


# Define input dimensions and parameters
batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]