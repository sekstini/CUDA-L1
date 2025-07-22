import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Define the CUDA kernel code
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// Fast math version of softplus with vectorized memory access
template <typename scalar_t>
__global__ void softplus_cuda_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    // Grid-stride loop for better GPU utilization
    const int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const scalar_t threshold = 20.0f;
    
    // Process elements in a grid-stride loop
    while (idx < size) {
        const scalar_t x = input[idx];
        
        // Numerically stable implementation with reduced branch divergence
        if (x > threshold) {
            output[idx] = x;
        } else if (x > 0.0f) {
            // For positive x, use log1p(exp(-x)) + x for better numerical stability
            output[idx] = x + __logf(1.0f + __expf(-x));
        } else {
            // For negative x, use log1p(exp(x))
            output[idx] = __logf(1.0f + __expf(x));
        }
        
        idx += stride;
    }
}

// Vectorized version for float4 (processes 4 elements at once)
__global__ void softplus_cuda_kernel_vec4(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const int size) {
    
    const int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    const float threshold = 20.0f;
    
    // Process elements in blocks of 4
    while (idx < size / 4) {
        const float4 x4 = input[idx];
        float4 result;
        
        // Process each component of float4 separately
        // x component
        if (x4.x > threshold) {
            result.x = x4.x;
        } else if (x4.x > 0.0f) {
            result.x = x4.x + __logf(1.0f + __expf(-x4.x));
        } else {
            result.x = __logf(1.0f + __expf(x4.x));
        }
        
        // y component
        if (x4.y > threshold) {
            result.y = x4.y;
        } else if (x4.y > 0.0f) {
            result.y = x4.y + __logf(1.0f + __expf(-x4.y));
        } else {
            result.y = __logf(1.0f + __expf(x4.y));
        }
        
        // z component
        if (x4.z > threshold) {
            result.z = x4.z;
        } else if (x4.z > 0.0f) {
            result.z = x4.z + __logf(1.0f + __expf(-x4.z));
        } else {
            result.z = __logf(1.0f + __expf(x4.z));
        }
        
        // w component
        if (x4.w > threshold) {
            result.w = x4.w;
        } else if (x4.w > 0.0f) {
            result.w = x4.w + __logf(1.0f + __expf(-x4.w));
        } else {
            result.w = __logf(1.0f + __expf(x4.w));
        }
        
        output[idx] = result;
        idx += stride;
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    // Optimize thread block size for element-wise operations
    const int threads = 256;
    
    // Calculate optimal grid size
    // Limit to 1024 blocks for small inputs to avoid excessive overhead
    const int max_blocks = 1024;
    const int blocks = min(max_blocks, (size + threads - 1) / threads);
    
    if (input.scalar_type() == torch::ScalarType::Float && size >= 1024 && size % 4 == 0) {
        // Use vectorized version for float tensors with size divisible by 4
        softplus_cuda_kernel_vec4<<<blocks, threads>>>(
            reinterpret_cast<float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            size);
    } else {
        // Use standard version for other cases
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "softplus_cuda_forward", ([&] {
            softplus_cuda_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                size);
        }));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
"""

# JIT compile the CUDA extension
try:
    softplus_cuda = load_inline(
        name="softplus_cuda",
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=["forward"],
        with_cuda=True,
        extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v"]
    )
except Exception as e:
    # Fallback to PyTorch implementation if compilation fails
    print(f"CUDA compilation failed: {e}")
    softplus_cuda = None

class ModelNew(nn.Module):
    """
    Simple model that performs a Softplus activation with optimized CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.use_cuda_kernel = softplus_cuda is not None and torch.cuda.is_available()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        if self.use_cuda_kernel and x.is_cuda:
            return softplus_cuda.forward(x)
        else:
            # Fallback to PyTorch implementation
            return torch.nn.functional.softplus(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed