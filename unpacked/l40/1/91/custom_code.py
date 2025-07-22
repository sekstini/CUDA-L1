import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Define CUDA kernel for reverse cumsum
cuda_source = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void reverse_cumsum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int seq_length) {
    
    // Get batch index
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Get pointers to the start of this batch's data
        const scalar_t* input_row = input + batch_idx * seq_length;
        scalar_t* output_row = output + batch_idx * seq_length;
        
        // Start from the end and work backwards
        scalar_t running_sum = 0;
        for (int i = seq_length - 1; i >= 0; i--) {
            running_sum += input_row[i];
            output_row[i] = running_sum;
        }
    }
}

torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim) {
    // Check if the input is contiguous, if not make it contiguous
    input = input.contiguous();
    
    // Get input shape
    auto shape = input.sizes();
    auto dtype = input.scalar_type();
    
    // Create output tensor
    auto output = torch::empty_like(input);
    
    // Handle only dim=1 for 2D tensors for now
    if (input.dim() == 2 && dim == 1) {
        const int batch_size = shape[0];
        const int seq_length = shape[1];
        
        // Launch kernel
        const int threads_per_block = 128;
        const int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
        
        AT_DISPATCH_FLOATING_TYPES(dtype, "reverse_cumsum_kernel", ([&] {
            reverse_cumsum_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                seq_length
            );
        }));
        
        return output;
    } else {
        // Fall back to PyTorch implementation for other cases
        return torch::cumsum(input.flip(dim), dim).flip(dim);
    }
}

torch::Tensor reverse_cumsum_forward(torch::Tensor input, int dim) {
    return reverse_cumsum_cuda(input, dim);
}

torch::Tensor reverse_cumsum_backward(torch::Tensor grad_output, int dim) {
    // For backward pass, we need to do a regular cumsum
    return torch::cumsum(grad_output, dim);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reverse_cumsum_forward, "Reverse Cumsum forward");
    m.def("backward", &reverse_cumsum_backward, "Reverse Cumsum backward");
}
'''

# Check if CUDA is available
if torch.cuda.is_available():
    try:
        # Attempt to load the extension
        reverse_cumsum_cuda = load_inline(
            name="reverse_cumsum_cuda",
            cpp_sources="",
            cuda_sources=cuda_source,
            functions=["forward", "backward"],
            with_cuda=True,
            extra_cuda_cflags=["-O3"]
        )
        
        class ReverseCumsumFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, dim):
                ctx.dim = dim
                return reverse_cumsum_cuda.forward(input, dim)
            
            @staticmethod
            def backward(ctx, grad_output):
                return reverse_cumsum_cuda.backward(grad_output, ctx.dim), None
        
        CUSTOM_KERNEL_AVAILABLE = True
    except Exception as e:
        print(f"Failed to load CUDA extension: {e}")
        CUSTOM_KERNEL_AVAILABLE = False
else:
    CUSTOM_KERNEL_AVAILABLE = False

class ModelNew(nn.Module):
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        if CUSTOM_KERNEL_AVAILABLE and x.is_cuda and x.dim() == 2 and self.dim == 1:
            # Use our custom CUDA kernel for the specific case
            return ReverseCumsumFunction.apply(x, self.dim)
        else:
            # Fall back to the reference implementation for other cases
            return torch.cumsum(x.flip(self.dim), dim=self.dim).flip(self.dim)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]