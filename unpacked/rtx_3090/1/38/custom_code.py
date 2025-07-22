import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel code
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void l1_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int dim) {
    
    // Each block processes one sample in the batch
    const int batch_idx = blockIdx.x;
    
    // Get pointers to the current batch's input and output
    const scalar_t* batch_input = input + batch_idx * dim;
    scalar_t* batch_output = output + batch_idx * dim;
    
    // Shared memory for reduction
    extern __shared__ float shared_mem[];
    
    // Calculate L1 norm using parallel reduction
    float thread_sum = 0.0f;
    
    // Each thread processes multiple elements with stride for better memory coalescing
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        thread_sum += fabsf(static_cast<float>(batch_input[i]));
    }
    
    // Store thread's sum in shared memory
    shared_mem[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // Perform parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed within a warp)
    if (threadIdx.x < 32) {
        // Use warp shuffle operations for faster reduction
        float sum = shared_mem[threadIdx.x];
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // Only the first thread in the warp has the final sum
        if (threadIdx.x == 0) {
            shared_mem[0] = sum;
        }
    }
    
    // Make sure the final sum is visible to all threads
    __syncthreads();
    
    // Get the final L1 norm (sum of absolute values)
    const float l1_norm = shared_mem[0];
    const float epsilon = 1e-10f; // Small epsilon to avoid division by zero
    
    // Pre-compute the inverse norm for faster division
    const float inv_norm = (l1_norm > epsilon) ? (1.0f / l1_norm) : 0.0f;
    
    // Normalize each element
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (l1_norm > epsilon) {
            batch_output[i] = static_cast<scalar_t>(batch_input[i] * inv_norm);
        } else {
            // Handle division by zero - match PyTorch's behavior
            batch_output[i] = batch_input[i];
        }
    }
}

torch::Tensor l1_norm_cuda_forward(torch::Tensor input) {
    const auto batch_size = input.size(0);
    const auto dim = input.size(1);
    
    // Create output tensor
    auto output = torch::empty_like(input);
    
    // Calculate block and grid dimensions
    const int threads_per_block = 256;  // Optimize based on GPU architecture
    const int blocks = batch_size;
    const int shared_mem_size = threads_per_block * sizeof(float);
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "l1_norm_forward_cuda", ([&] {
        l1_norm_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            dim
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &l1_norm_cuda_forward, "L1 Norm forward (CUDA)");
}
"""

# Try to load the CUDA extension
try:
    l1_norm_cuda = load_inline(
        name="l1_norm_cuda",
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=["forward"],
        with_cuda=True,
        extra_cuda_cflags=["-O3", "--use_fast_math"]
    )
    
    CUDA_EXTENSION_AVAILABLE = True
except Exception as e:
    print(f"CUDA extension could not be loaded, falling back to PyTorch implementation: {e}")
    CUDA_EXTENSION_AVAILABLE = False

class ModelNew(nn.Module):
    """
    Simple model that performs L1 normalization with optimized implementation.
    """
    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim, ...).

        Returns:
            torch.Tensor: Output tensor with L1 normalization applied, same shape as input.
        """
        # Check if we can use our custom CUDA kernel
        if CUDA_EXTENSION_AVAILABLE and x.is_cuda and x.dim() == 2:
            # Make sure input is contiguous for optimal performance
            if not x.is_contiguous():
                x = x.contiguous()
            return l1_norm_cuda.forward(x)
        
        # Fallback to optimized PyTorch implementation
        # Using torch.linalg.vector_norm which is more optimized than torch.sum(torch.abs())
        return x / torch.linalg.vector_norm(x, ord=1, dim=1, keepdim=True)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []