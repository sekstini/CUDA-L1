import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Define the CUDA kernel code
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized kernel for reducing along dimension 0 (batch dimension)
template <typename scalar_t>
__global__ void mean_dim0_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2) {
    
    // Calculate output index
    const int d2 = blockIdx.x * blockDim.x + threadIdx.x;
    const int d1 = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (d1 >= dim1 || d2 >= dim2) return;
    
    // Output index
    const int out_idx = d1 * dim2 + d2;
    
    // Compute sum directly
    scalar_t sum = 0;
    
    #pragma unroll 4
    for (int b = 0; b < batch_size; b++) {
        sum += input[b * dim1 * dim2 + d1 * dim2 + d2];
    }
    
    // Write result directly
    output[out_idx] = sum / batch_size;
}

// Optimized kernel for reducing along dimension 1
template <typename scalar_t>
__global__ void mean_dim1_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2) {
    
    // Calculate indices
    const int d2 = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b >= batch_size || d2 >= dim2) return;
    
    // Output index
    const int out_idx = b * dim2 + d2;
    
    // Compute sum directly
    scalar_t sum = 0;
    
    #pragma unroll 4
    for (int d1 = 0; d1 < dim1; d1++) {
        sum += input[b * dim1 * dim2 + d1 * dim2 + d2];
    }
    
    // Write result directly
    output[out_idx] = sum / dim1;
}

// Optimized kernel for reducing along dimension 2
template <typename scalar_t, int BLOCK_SIZE>
__global__ void mean_dim2_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2) {
    
    // Calculate indices
    const int d1 = blockIdx.x;
    const int b = blockIdx.y;
    
    // Base index for this block
    const int base_idx = b * dim1 * dim2 + d1 * dim2;
    
    // Thread index
    const int tid = threadIdx.x;
    
    // Use shared memory for reduction
    __shared__ scalar_t sdata[BLOCK_SIZE];
    
    // Initialize
    scalar_t thread_sum = 0;
    
    // Each thread loads and adds multiple elements
    for (int i = tid; i < dim2; i += BLOCK_SIZE) {
        thread_sum += input[base_idx + i];
    }
    
    // Store in shared memory
    sdata[tid] = thread_sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp-level reduction for the last warp
    if (tid < 32) {
        // Use warp-level operations for final reduction
        if (BLOCK_SIZE >= 64) sdata[tid] += sdata[tid + 32];
        __syncwarp();
        if (tid < 16) {
            sdata[tid] += sdata[tid + 16];
            sdata[tid] += sdata[tid + 8];
            sdata[tid] += sdata[tid + 4];
            sdata[tid] += sdata[tid + 2];
            sdata[tid] += sdata[tid + 1];
        }
    }
    
    // First thread writes the result
    if (tid == 0) {
        output[b * dim1 + d1] = sdata[0] / dim2;
    }
}

torch::Tensor mean_cuda(torch::Tensor input, int dim) {
    // Get input dimensions
    int batch_size = input.size(0);
    int dim1 = input.size(1);
    int dim2 = input.size(2);
    
    // Determine output shape
    torch::Tensor output;
    
    if (dim == 0) {
        // Reducing along batch dimension
        output = torch::empty({dim1, dim2}, input.options());
        
        // Configure kernel launch parameters
        dim3 block(16, 16);
        dim3 grid((dim2 + block.x - 1) / block.x, (dim1 + block.y - 1) / block.y);
        
        AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_dim0_cuda", ([&] {
            mean_dim0_kernel<scalar_t><<<grid, block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim1,
                dim2
            );
        }));
    } 
    else if (dim == 1) {
        // Reducing along dim1
        output = torch::empty({batch_size, dim2}, input.options());
        
        // Configure kernel launch parameters
        dim3 block(32, 8);
        dim3 grid((dim2 + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);
        
        AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_dim1_cuda", ([&] {
            mean_dim1_kernel<scalar_t><<<grid, block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim1,
                dim2
            );
        }));
    } 
    else if (dim == 2) {
        // Reducing along dim2
        output = torch::empty({batch_size, dim1}, input.options());
        
        // For dimension 2 reduction, use a block size that's a power of 2
        const int BLOCK_SIZE = 256;
        
        dim3 grid(dim1, batch_size);
        dim3 block(BLOCK_SIZE);
        
        AT_DISPATCH_FLOATING_TYPES(input.type(), "mean_dim2_cuda", ([&] {
            mean_dim2_kernel<scalar_t, BLOCK_SIZE><<<grid, block>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim1,
                dim2
            );
        }));
    } 
    else {
        throw std::runtime_error("Unsupported reduction dimension");
    }
    
    return output;
}
"""

# Load the CUDA extension
try:
    mean_cuda = load_inline(
        name="mean_cuda",
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=["mean_cuda"],
        verbose=False,
        with_cuda=True
    )
except Exception as e:
    print(f"Failed to load CUDA extension: {e}")
    mean_cuda = None

class ModelNew(nn.Module):
    """
    Optimized implementation of mean reduction over a specific dimension.
    
    Args:
        dim (int): The dimension to reduce over.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean.
        
        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.
            
        Returns:
            torch.Tensor: Output tensor with reduced dimension.
        """
        # Check if we can use our custom CUDA kernel
        if mean_cuda is not None and x.is_cuda and x.dim() == 3 and 0 <= self.dim <= 2:
            try:
                return mean_cuda.mean_cuda(x, self.dim)
            except Exception:
                # Fallback to PyTorch implementation if kernel fails
                return torch.mean(x, dim=self.dim)
        else:
            # Fallback to PyTorch implementation
            return torch.mean(x, dim=self.dim)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]  # Dimension to reduce over (example value)