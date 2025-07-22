import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused Mish + Add + Hardtanh + Scale operations
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t mish(scalar_t x) {
    // Numerically stable Mish implementation: x * tanh(softplus(x))
    return x * tanh(log1p(exp(x)));
}

// Specialized kernel optimized for the exact output dimensions (32x32)
template <typename scalar_t>
__global__ void fused_post_conv_kernel_32x32(
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const scalar_t add_value,
    const scalar_t scale) {
    
    // Using a 32x8 thread block configuration (256 threads)
    const int x = threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Calculate stride for each dimension
    const int stride_batch = channels * height * width;
    const int stride_channel = height * width;
    
    // Each thread processes elements for all batches and channels at this (x,y) location
    if (x < width && y < height) {
        for (int n = blockIdx.x; n < batch_size; n += gridDim.x) {
            #pragma unroll 4
            for (int c = 0; c < channels; c++) {
                const int idx = n * stride_batch + c * stride_channel + y * width + x;
                
                // Load from global memory once
                scalar_t val = output[idx];
                
                // Apply Mish: x * tanh(softplus(x))
                val = mish(val);
                
                // Add constant value
                val += add_value;
                
                // Apply Hardtanh
                val = max(scalar_t(-1.0), min(scalar_t(1.0), val));
                
                // Scale
                val *= scale;
                
                // Write back to output (in-place operation)
                output[idx] = val;
            }
        }
    }
}

// Vectorized kernel for batch processing
template <typename scalar_t>
__global__ void fused_post_conv_kernel_vectorized(
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const scalar_t add_value,
    const scalar_t scale) {
    
    // 2D block for better mapping to 2D output
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Calculate stride for each dimension
        const int stride_batch = channels * height * width;
        const int stride_channel = height * width;
        const int pos = y * width + x;
        
        // Process 4 batches at a time when possible
        int n = 0;
        for (; n < batch_size - 3; n += 4) {
            for (int c = 0; c < channels; c++) {
                const int base_idx = n * stride_batch + c * stride_channel + pos;
                
                // Process 4 batch elements at once
                scalar_t vals[4];
                
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    vals[i] = output[base_idx + i * stride_batch];
                    vals[i] = mish(vals[i]);
                    vals[i] += add_value;
                    vals[i] = max(scalar_t(-1.0), min(scalar_t(1.0), vals[i]));
                    vals[i] *= scale;
                    output[base_idx + i * stride_batch] = vals[i];
                }
            }
        }
        
        // Process remaining batches
        for (; n < batch_size; n++) {
            for (int c = 0; c < channels; c++) {
                const int idx = n * stride_batch + c * stride_channel + pos;
                
                scalar_t val = output[idx];
                val = mish(val);
                val += add_value;
                val = max(scalar_t(-1.0), min(scalar_t(1.0), val));
                val *= scale;
                output[idx] = val;
            }
        }
    }
}

// Generic kernel for other dimensions
template <typename scalar_t>
__global__ void fused_post_conv_kernel_generic(
    scalar_t* __restrict__ output,
    const int numel,
    const scalar_t add_value,
    const scalar_t scale) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements with a stride
    for (int i = idx; i < numel; i += stride) {
        scalar_t val = output[i];
        
        // Apply Mish: x * tanh(softplus(x))
        val = mish(val);
        
        // Add constant value
        val += add_value;
        
        // Apply Hardtanh
        val = max(scalar_t(-1.0), min(scalar_t(1.0), val));
        
        // Scale
        val *= scale;
        
        // Write back to output (in-place operation)
        output[i] = val;
    }
}

void fused_post_conv_cuda(
    torch::Tensor output,
    float add_value,
    float scale) {
    
    const int batch_size = output.size(0);
    const int channels = output.size(1);
    const int height = output.size(2);
    const int width = output.size(3);
    const int numel = output.numel();
    
    // Choose the appropriate kernel based on dimensions
    if (height == 32 && width == 32) {
        // Specialized kernel for 32x32 dimensions (output size after convolution)
        const dim3 threads(32, 8);  // 32x8 = 256 threads per block
        const dim3 blocks(
            min(32, batch_size),  // Process multiple batches in parallel
            (height + threads.y - 1) / threads.y
        );
        
        AT_DISPATCH_FLOATING_TYPES(output.type(), "fused_post_conv_cuda", ([&] {
            fused_post_conv_kernel_32x32<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                batch_size,
                channels,
                height,
                width,
                static_cast<scalar_t>(add_value),
                static_cast<scalar_t>(scale));
        }));
    }
    else if (batch_size >= 4 && height <= 64 && width <= 64) {
        // Use vectorized kernel for small to medium dimensions
        const dim3 threads(16, 16);
        const dim3 blocks(
            (width + threads.x - 1) / threads.x,
            (height + threads.y - 1) / threads.y
        );
        
        AT_DISPATCH_FLOATING_TYPES(output.type(), "fused_post_conv_cuda", ([&] {
            fused_post_conv_kernel_vectorized<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                batch_size,
                channels,
                height,
                width,
                static_cast<scalar_t>(add_value),
                static_cast<scalar_t>(scale));
        }));
    }
    else {
        // Generic kernel for other dimensions
        const int threads = 256;
        const int blocks = min(65535, (numel + threads - 1) / threads);
        
        AT_DISPATCH_FLOATING_TYPES(output.type(), "fused_post_conv_cuda", ([&] {
            fused_post_conv_kernel_generic<scalar_t><<<blocks, threads>>>(
                output.data_ptr<scalar_t>(),
                numel,
                static_cast<scalar_t>(add_value),
                static_cast<scalar_t>(scale));
        }));
    }
}
"""

cpp_source = """
#include <torch/extension.h>

// CUDA forward declaration
void fused_post_conv_cuda(
    torch::Tensor output,
    float add_value,
    float scale);

// C++ interface
void fused_post_conv(
    torch::Tensor output,
    float add_value,
    float scale) {
    
    fused_post_conv_cuda(output, add_value, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Fused post-convolution operations");
}
"""

# Try to load the CUDA extension
try:
    fused_ops = load_inline(
        name="fused_ops",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=["fused_post_conv"],
        with_cuda=True,
        extra_cuda_cflags=["-O3", "--use_fast_math"]
    )
    has_cuda_extension = True
except Exception as e:
    print(f"Could not load CUDA extension: {e}")
    has_cuda_extension = False

class ModelNew(nn.Module):
    """
    Optimized implementation of the model that performs a transposed convolution,
    applies Mish activation, adds a value, applies Hardtanh activation, and scales the output.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolving kernel
        stride (int): Stride of the convolution
        padding (int): Padding added to all sides of the input
        output_padding (int): Additional size added to one side of the output
        add_value (float): Value to add after Mish activation
        scale (float): Scaling factor to apply after Hardtanh
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        
        # Create a dedicated CUDA stream if CUDA is available
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
    
    def forward(self, x):
        # Use a dedicated CUDA stream if available
        with torch.cuda.stream(self.stream) if self.stream is not None and x.is_cuda else torch.no_grad():
            # Step 1: Transposed convolution
            x = self.conv_transpose(x)
            
            # Steps 2-5: Fused operations (Mish + Add + Hardtanh + Scale)
            if x.is_cuda and has_cuda_extension:
                try:
                    # Use our optimized implementation with custom CUDA kernel
                    fused_ops.fused_post_conv(x, self.add_value, self.scale)
                    return x  # Operations are performed in-place
                except Exception as e:
                    print(f"Error in optimized forward pass: {e}")
                    # Fall back to standard implementation
            
            # Standard PyTorch implementation (fallback)
            x = F.mish(x)
            x = x + self.add_value
            x = F.hardtanh(x, min_val=-1, max_val=1)
            x = x * self.scale
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]