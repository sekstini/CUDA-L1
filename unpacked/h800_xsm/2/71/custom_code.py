import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import tempfile
from torch.utils.cpp_extension import load

# Define CUDA kernel code
CUDA_SOURCE = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void fused_conv2d_div_leakyrelu_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size,
    float divisor,
    float negative_slope) {
    
    // Calculate output position
    const int n = blockIdx.z;
    const int c = blockIdx.y;
    const int h = blockIdx.x / out_width;
    const int w = blockIdx.x % out_width;
    
    if (n >= batch_size || c >= out_channels || h >= out_height || w >= out_width)
        return;
    
    // Initialize output value
    scalar_t result = bias ? bias[c] : 0.0f;
    
    // Perform convolution
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int ih = h + kh;
                const int iw = w + kw;
                
                if (ih < in_height && iw < in_width) {
                    const int input_idx = ((n * in_channels + ic) * in_height + ih) * in_width + iw;
                    const int weight_idx = ((c * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    
                    result += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Apply division
    result /= divisor;
    
    // Apply LeakyReLU
    if (result < 0) {
        result *= negative_slope;
    }
    
    // Store result
    const int output_idx = ((n * out_channels + c) * out_height + h) * out_width + w;
    output[output_idx] = result;
}

std::vector<torch::Tensor> fused_conv2d_div_leakyrelu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor,
    float negative_slope) {
    
    // Get dimensions
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_height = input.size(2);
    const auto in_width = input.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);
    const auto out_height = in_height - kernel_size + 1;
    const auto out_width = in_width - kernel_size + 1;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                              input.options());
    
    // Calculate grid and block dimensions
    const dim3 blocks(out_height * out_width, out_channels, batch_size);
    const dim3 threads(1, 1, 1);
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_conv2d_div_leakyrelu_cuda", ([&] {
        fused_conv2d_div_leakyrelu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            out_height,
            out_width,
            kernel_size,
            divisor,
            negative_slope
        );
    }));
    
    return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_conv2d_div_leakyrelu_cuda, "Fused Conv2d+Div+LeakyReLU forward (CUDA)");
}
"""

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, divides by a constant, and applies LeakyReLU.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        divisor (float): Divisor value for the division operation
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.divisor = divisor
        self.negative_slope = 0.01
        
        # Create standard Conv2d layer with same parameters as reference
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # Pre-divide weights and bias for fallback implementation
        self.optimized_weight = nn.Parameter(self.conv.weight.data.clone() / self.divisor)
        self.optimized_bias = nn.Parameter(self.conv.bias.data.clone() / self.divisor) if self.conv.bias is not None else None
        
        # Enable cuDNN optimizations for fallback path
        torch.backends.cudnn.benchmark = True
        
        # Compile CUDA extension
        self.fused_kernel = None
        self.use_fallback = False
        
        # Try to compile the CUDA extension
        try:
            # Create a unique name for the extension to avoid conflicts
            extension_name = f"fused_conv2d_div_leakyrelu_{id(self)}"
            
            # Write CUDA code to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.cu', delete=False) as f:
                f.write(CUDA_SOURCE.encode('utf-8'))
                cuda_file = f.name
            
            # Compile the extension
            self.fused_kernel = load(
                name=extension_name,
                sources=[cuda_file],
                verbose=False
            )
            
            # Clean up temporary file
            os.unlink(cuda_file)
        except Exception as e:
            print(f"Failed to compile CUDA extension: {e}")
            self.use_fallback = True
    
    def forward(self, x):
        # Try to use the fused CUDA kernel if available
        if not self.use_fallback and self.fused_kernel is not None and x.is_cuda:
            try:
                # Ensure inputs are contiguous
                x = x.contiguous()
                weight = self.conv.weight.contiguous()
                bias = self.conv.bias.contiguous() if self.conv.bias is not None else None
                
                # Call the fused kernel
                return self.fused_kernel.forward(x, weight, bias, self.divisor, self.negative_slope)[0]
            except Exception as e:
                print(f"CUDA kernel failed, using fallback: {e}")
                self.use_fallback = True
        
        # Fallback implementation using pre-divided weights
        x = F.conv2d(x, self.optimized_weight, self.optimized_bias)
        return F.leaky_relu(x, negative_slope=self.negative_slope, inplace=True)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
divisor = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]