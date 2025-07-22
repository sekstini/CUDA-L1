import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# SELU constants
ALPHA = 1.6732632423543772848170429916717
SCALE = 1.0507009873554804934193349852946

# Define the CUDA kernel code
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// SELU constants
#define ALPHA 1.6732632423543772848170429916717f
#define SCALE 1.0507009873554804934193349852946f

template <typename scalar_t>
__global__ void selu_kernel(const scalar_t* __restrict__ input, 
                           scalar_t* __restrict__ output, 
                           const int size) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop for handling large tensors
    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        const scalar_t x = input[i];
        // Compute SELU: scale * x if x > 0, scale * alpha * (exp(x) - 1) if x <= 0
        // Use branchless technique to minimize divergence
        const bool is_positive = x > 0;
        const scalar_t exp_val = is_positive ? 0 : expf(x) - 1.0f;
        const scalar_t scale_val = is_positive ? SCALE : SCALE * ALPHA;
        const scalar_t val = is_positive ? x : exp_val;
        output[i] = scale_val * val;
    }
}

// Vectorized version using float4 for better memory throughput
__global__ void selu_kernel_vec4(const float4* __restrict__ input, 
                                float4* __restrict__ output, 
                                const int size) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride loop for handling large tensors
    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        const float4 x4 = input[i];
        float4 result;
        
        // Process each element of the float4 using branchless technique
        // Element x
        {
            const bool is_positive = x4.x > 0;
            const float exp_val = is_positive ? 0 : expf(x4.x) - 1.0f;
            const float scale_val = is_positive ? SCALE : SCALE * ALPHA;
            const float val = is_positive ? x4.x : exp_val;
            result.x = scale_val * val;
        }
        
        // Element y
        {
            const bool is_positive = x4.y > 0;
            const float exp_val = is_positive ? 0 : expf(x4.y) - 1.0f;
            const float scale_val = is_positive ? SCALE : SCALE * ALPHA;
            const float val = is_positive ? x4.y : exp_val;
            result.y = scale_val * val;
        }
        
        // Element z
        {
            const bool is_positive = x4.z > 0;
            const float exp_val = is_positive ? 0 : expf(x4.z) - 1.0f;
            const float scale_val = is_positive ? SCALE : SCALE * ALPHA;
            const float val = is_positive ? x4.z : exp_val;
            result.z = scale_val * val;
        }
        
        // Element w
        {
            const bool is_positive = x4.w > 0;
            const float exp_val = is_positive ? 0 : expf(x4.w) - 1.0f;
            const float scale_val = is_positive ? SCALE : SCALE * ALPHA;
            const float val = is_positive ? x4.w : exp_val;
            result.w = scale_val * val;
        }
        
        output[i] = result;
    }
}

torch::Tensor selu_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    
    // Optimize thread/block configuration
    const int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    // Limit the number of blocks to avoid excessive overhead
    // For our specific dimensions (16*16384 = 262144 elements), this is appropriate
    if (blocks > 1024) blocks = 1024;
    
    // Check if we can use vectorized version (size must be divisible by 4 and data type must be float)
    if (input.scalar_type() == at::ScalarType::Float && size % 4 == 0 && 
        input.is_contiguous() && output.is_contiguous()) {
        
        // Use vectorized kernel for better memory throughput
        int vec_size = size / 4;
        selu_kernel_vec4<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            vec_size
        );
    } else {
        // Use standard kernel for all other cases
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_cuda", ([&] {
            selu_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                size
            );
        }));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_cuda, "SELU forward (CUDA)");
}
"""

# Try to load the CUDA extension with proper error handling
selu_cuda_ext = None
if torch.cuda.is_available():
    try:
        selu_cuda_ext = load_inline(
            name="selu_cuda_ext",
            cpp_sources="",
            cuda_sources=cuda_source,
            functions=["forward"],
            with_cuda=True,
            extra_cuda_cflags=["-O3", "--use_fast_math"]
        )
    except Exception as e:
        print(f"CUDA extension could not be loaded: {e}")

class ModelNew(nn.Module):
    """
    Simple model that performs a SELU activation with optimized CUDA implementation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        # Use our custom CUDA kernel if available and input is on CUDA
        if selu_cuda_ext is not None and x.is_cuda:
            # Move tensor to contiguous memory layout if needed
            if not x.is_contiguous():
                x = x.contiguous()
            return selu_cuda_ext.forward(x)
        else:
            # Fall back to PyTorch implementation
            return torch.selu(x)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed