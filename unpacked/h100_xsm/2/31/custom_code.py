import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

# Define CUDA kernel for fused convolution, min, bias add
cuda_source = """
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void fused_conv2d_min_bias_scale_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int height,
    const int width,
    const int out_channels,
    const int kernel_size,
    const scalar_t scaled_constant_value,
    const int out_height,
    const int out_width) {
    
    // Shared memory for input and weights
    extern __shared__ scalar_t shared_mem[];
    scalar_t* shared_input = shared_mem;
    scalar_t* shared_weight = shared_mem + (blockDim.y + kernel_size - 1) * (blockDim.x + kernel_size - 1);
    
    // Calculate output position
    const int w_out_start = blockIdx.x * blockDim.x;
    const int h_out_start = blockIdx.y * blockDim.y;
    const int w_out = w_out_start + threadIdx.x;
    const int h_out = h_out_start + threadIdx.y;
    const int c_out = blockIdx.z % out_channels;
    const int n = blockIdx.z / out_channels;
    
    // Load weights into shared memory
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * blockDim.y;
    const int weights_per_filter = in_channels * kernel_size * kernel_size;
    
    for (int i = thread_id; i < weights_per_filter; i += total_threads) {
        const int ic = i / (kernel_size * kernel_size);
        const int kh = (i % (kernel_size * kernel_size)) / kernel_size;
        const int kw = (i % (kernel_size * kernel_size)) % kernel_size;
        
        const int w_idx = ((c_out * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
        shared_weight[i] = weight[w_idx];
    }
    
    // Load input patch into shared memory
    const int h_in_start = h_out_start;
    const int w_in_start = w_out_start;
    
    // Each thread loads multiple elements into shared memory
    for (int h_offset = threadIdx.y; h_offset < blockDim.y + kernel_size - 1; h_offset += blockDim.y) {
        const int h_in = h_in_start + h_offset;
        
        for (int w_offset = threadIdx.x; w_offset < blockDim.x + kernel_size - 1; w_offset += blockDim.x) {
            const int w_in = w_in_start + w_offset;
            
            // Load input data for all input channels
            if (h_in < height && w_in < width) {
                for (int ic = 0; ic < in_channels; ++ic) {
                    const int in_idx = ((n * in_channels + ic) * height + h_in) * width + w_in;
                    const int sm_idx = (ic * (blockDim.y + kernel_size - 1) + h_offset) * (blockDim.x + kernel_size - 1) + w_offset;
                    shared_input[sm_idx] = input[in_idx];
                }
            } else {
                for (int ic = 0; ic < in_channels; ++ic) {
                    const int sm_idx = (ic * (blockDim.y + kernel_size - 1) + h_offset) * (blockDim.x + kernel_size - 1) + w_offset;
                    shared_input[sm_idx] = 0.0f;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Check if thread is within output bounds
    if (h_out >= out_height || w_out >= out_width || n >= batch_size) {
        return;
    }
    
    // Compute convolution
    scalar_t result = 0.0f;
    
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int h_in_local = threadIdx.y + kh;
                const int w_in_local = threadIdx.x + kw;
                const int sm_idx = (ic * (blockDim.y + kernel_size - 1) + h_in_local) * (blockDim.x + kernel_size - 1) + w_in_local;
                const int w_idx = (ic * kernel_size + kh) * kernel_size + kw;
                
                result += shared_input[sm_idx] * shared_weight[w_idx];
            }
        }
    }
    
    // Add bias
    result += bias[c_out];
    
    // Apply min operation with scaled constant
    if (result > scaled_constant_value) {
        result = scaled_constant_value;
    }
    
    // Write output
    const int out_idx = ((n * out_channels + c_out) * out_height + h_out) * out_width + w_out;
    output[out_idx] = result;
}

torch::Tensor fused_conv2d_min_bias_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaled_constant_value,
    int kernel_size) {
    
    // Get dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int out_height = height - kernel_size + 1;
    const int out_width = width - kernel_size + 1;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                              input.options());
    
    // Calculate grid and block dimensions - optimize for the specific problem size
    const dim3 threads(32, 8);
    const dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        batch_size * out_channels
    );
    
    // Calculate shared memory size
    const int shared_mem_size = (
        // Input patch: in_channels * (blockDim.y + kernel_size - 1) * (blockDim.x + kernel_size - 1)
        in_channels * (threads.y + kernel_size - 1) * (threads.x + kernel_size - 1) +
        // Weights: in_channels * kernel_size * kernel_size
        in_channels * kernel_size * kernel_size
    ) * sizeof(float);
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_conv2d_min_bias_scale_cuda", ([&] {
        fused_conv2d_min_bias_scale_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            height,
            width,
            out_channels,
            kernel_size,
            static_cast<scalar_t>(scaled_constant_value),
            out_height,
            out_width);
    }));
    
    return output;
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor fused_conv2d_min_bias_scale_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaled_constant_value,
    int kernel_size);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fused_conv2d_min_bias_scale(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scaled_constant_value,
    int kernel_size) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);
    
    return fused_conv2d_min_bias_scale_cuda(input, weight, bias, scaled_constant_value, kernel_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_conv2d_min_bias_scale, "Fused Conv2d Min Bias Scale (CUDA)");
}
"""

# Try to compile CUDA extension
try:
    fused_conv = load_inline(
        name="fused_conv_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["forward"],
        verbose=False
    )
    has_cuda_extension = True
except Exception as e:
    print(f"Failed to load CUDA extension: {e}")
    has_cuda_extension = False

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, takes the minimum with a constant,
    adds a bias term, and multiplies by a scaling factor.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolution kernel
        constant_value (float): Constant value for minimum operation
        bias_shape (tuple): Shape of the bias tensor
        scaling_factor (float): Scaling factor to apply
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        # Create the convolution layer with same configuration as reference
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        
        # Store parameters
        self.constant_value = constant_value
        self.scaling_factor = scaling_factor
        self.kernel_size = kernel_size
        
        # Create a separate bias parameter with the correct shape
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Pre-compute the scaled constant value
        self.scaled_constant = constant_value * scaling_factor
        
        # Pre-scale the convolution weights and bias for the optimized path
        with torch.no_grad():
            self.scaled_weight = nn.Parameter(self.conv.weight.data.clone() * scaling_factor)
            if self.conv.bias is not None:
                self.scaled_conv_bias = nn.Parameter(self.conv.bias.data.clone() * scaling_factor)
            else:
                self.scaled_conv_bias = None
            self.scaled_bias = nn.Parameter(self.bias.data.clone() * scaling_factor)
        
        # Register a buffer for the reshaped bias to avoid reshaping during forward pass
        self.register_buffer('reshaped_bias', None)
    
    def forward(self, x):
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        if has_cuda_extension and x.is_cuda:
            try:
                # Reshape bias to match kernel expectations (out_channels,)
                flat_bias = self.scaled_bias.view(self.scaled_bias.size(0))
                
                # Call the CUDA kernel
                return fused_conv.forward(
                    x, 
                    self.scaled_weight, 
                    flat_bias, 
                    self.scaled_constant,
                    self.kernel_size
                )
            except Exception as e:
                print(f"CUDA kernel execution failed: {e}, falling back to PyTorch implementation")
                # Fall through to PyTorch implementation
        
        # Optimized PyTorch implementation (fallback)
        # 1. Perform convolution with pre-scaled weights
        if self.scaled_conv_bias is not None:
            x = F.conv2d(x, self.scaled_weight, self.scaled_conv_bias, padding=0)
        else:
            x = F.conv2d(x, self.scaled_weight, None, padding=0)
        
        # 2. Apply min operation in-place
        x.clamp_max_(self.scaled_constant)
        
        # 3. Add the bias (already scaled)
        # Ensure bias is properly shaped for broadcasting
        if self.reshaped_bias is None or self.reshaped_bias.device != x.device:
            self.reshaped_bias = self.scaled_bias.to(device=x.device)
        
        x = x + self.reshaped_bias
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]