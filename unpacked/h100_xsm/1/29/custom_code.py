import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Define the CUDA kernel code
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void softplus_cuda_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    size_t size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        const scalar_t x = input[idx];
        
        // Numerically stable implementation of softplus
        if (x > 20.0) {
            // For large x, softplus(x) ≈ x to avoid overflow
            output[idx] = x;
        } else if (x < -20.0) {
            // For very negative x, softplus(x) ≈ exp(x) to avoid underflow
            output[idx] = exp(x);
        } else {
            // Standard formula with improved numerical stability
            output[idx] = log1p(exp(x));
        }
    }
}

torch::Tensor softplus_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_cuda", ([&] {
        softplus_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor softplus_cuda(torch::Tensor input);

torch::Tensor softplus(torch::Tensor input) {
    return softplus_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softplus", &softplus, "Optimized softplus implementation");
}
"""

# Compile the CUDA extension
try:
    softplus_cuda = load_inline(
        name="softplus_cuda",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["softplus"],
        verbose=True,
        build_directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"),
    )
except Exception as e:
    print(f"Failed to compile CUDA extension: {e}")
    # Fallback to using PyTorch's implementation
    softplus_cuda = None

class ModelNew(nn.Module):
    """
    Optimized model that performs a Softplus activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation to the input tensor using optimized CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        if softplus_cuda is not None and x.is_cuda:
            return softplus_cuda.softplus(x)
        else:
            # Fallback to PyTorch implementation if CUDA extension fails
            return torch.nn.functional.softplus(x)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed