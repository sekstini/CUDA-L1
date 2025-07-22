import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Define the CUDA kernel for HardSigmoid
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void hardsigmoid_cuda_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    size_t size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements for better efficiency
    for (int i = idx; i < size; i += stride) {
        const scalar_t val = input[i];
        // Direct computation of HardSigmoid: min(max(x + 3, 0), 6) / 6
        output[i] = min(max(val + 3.0f, 0.0f), 6.0f) / 6.0f;
    }
}

torch::Tensor hardsigmoid_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    
    // Calculate optimal number of blocks based on tensor size
    int64_t total_elements = input.numel();
    int blocks = (total_elements + threads - 1) / threads;
    
    // Limit the number of blocks to avoid excessive overhead
    blocks = std::min(blocks, 1024);
    
    // Launch kernel based on data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "hardsigmoid_cuda", ([&] {
        hardsigmoid_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total_elements
        );
    }));
    
    return output;
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor hardsigmoid_cuda(torch::Tensor input);

torch::Tensor hardsigmoid(torch::Tensor input) {
    return hardsigmoid_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hardsigmoid", &hardsigmoid, "HardSigmoid function");
}
"""

# Compile the CUDA extension
hardsigmoid_cuda = None
try:
    # Create a build directory with a unique name to avoid conflicts
    build_dir = "_cuda_hardsigmoid_build"
    os.makedirs(build_dir, exist_ok=True)
    
    hardsigmoid_cuda = load_inline(
        name="hardsigmoid_cuda_opt",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["hardsigmoid"],
        verbose=False,
        build_directory=build_dir,
        with_cuda=True
    )
except Exception as e:
    print(f"CUDA extension compilation failed: {e}")
    hardsigmoid_cuda = None

class ModelNew(nn.Module):
    """
    Optimized model that performs a HardSigmoid activation using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardSigmoid activation to the input tensor using a custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
        """
        # Use our custom CUDA kernel if available and input is on GPU
        if hardsigmoid_cuda is not None and torch.cuda.is_available():
            # Move tensor to GPU if it's not already there
            if not x.is_cuda:
                x = x.cuda()
            return hardsigmoid_cuda.hardsigmoid(x)
        else:
            # Fallback to PyTorch's implementation
            return torch.nn.functional.hardsigmoid(x)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed