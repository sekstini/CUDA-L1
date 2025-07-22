import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

# Define CUDA kernel for fused bias_add + scale + sigmoid
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_bias_scale_sigmoid_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ scale,
    const int N,
    const int C,
    const int H,
    const int W) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;
    
    const int c = (idx / (H * W)) % C;
    
    // Apply bias, scale, and sigmoid in one operation
    scalar_t val = input[idx];
    val = val + bias[c];
    val = val * scale[c];
    val = 1.0f / (1.0f + expf(-val));
    
    output[idx] = val;
}

torch::Tensor fused_bias_scale_sigmoid_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    torch::Tensor scale) {
    
    auto output = torch::empty_like(input);
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    
    const int threads = 256;
    const int blocks = (N * C * H * W + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_bias_scale_sigmoid_cuda", ([&] {
        fused_bias_scale_sigmoid_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            scale.data_ptr<scalar_t>(),
            N, C, H, W);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_bias_scale_sigmoid", &fused_bias_scale_sigmoid_cuda, "Fused Bias + Scale + Sigmoid (CUDA)");
}
"""

# Try to load the custom CUDA extension
try:
    fused_ops = load_inline(
        name="fused_ops",
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=["fused_bias_scale_sigmoid"],
        with_cuda=True,
        extra_cuda_cflags=["-O3"]
    )
except Exception as e:
    print(f"Warning: Could not load CUDA extension: {e}")
    fused_ops = None

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, adds a bias term, scales, applies sigmoid, 
    and performs group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        
        # Flag to determine if we can use our optimized kernel
        self.use_optimized = torch.cuda.is_available() and fused_ops is not None

    def forward(self, x):
        # Use PyTorch's optimized Conv2d implementation
        x = self.conv(x)
        
        if self.use_optimized and x.is_cuda:
            try:
                # Use our fused kernel for bias + scale + sigmoid
                bias_flat = self.bias.view(-1)
                scale_flat = self.scale.view(-1)
                x = fused_ops.fused_bias_scale_sigmoid(x, bias_flat, scale_flat)
            except Exception as e:
                # Fallback to standard implementation if there's an error
                print(f"Warning: Fused kernel failed, falling back to standard implementation: {e}")
                x = x + self.bias
                x = x * self.scale
                x = torch.sigmoid(x)
        else:
            # Standard implementation
            x = x + self.bias
            x = x * self.scale
            x = torch.sigmoid(x)
        
        # Apply group normalization
        x = self.group_norm(x)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]