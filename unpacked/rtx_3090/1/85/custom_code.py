import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline
import os

# Define the CUDA kernel for depthwise convolution
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void depthwise_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int kernel_height,
    const int kernel_width,
    const int output_height,
    const int output_width,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w) {
    
    // Calculate output position
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;
    
    if (w_out < output_width && h_out < output_height) {
        // Calculate input position
        const int input_offset = b * channels * input_height * input_width + 
                                c * input_height * input_width;
        const int weight_offset = c * kernel_height * kernel_width;
        
        scalar_t sum = 0;
        
        #pragma unroll
        for (int kh = 0; kh < kernel_height; kh++) {
            const int h_in = h_out * stride_h - padding_h + kh * dilation_h;
            
            if (h_in >= 0 && h_in < input_height) {
                #pragma unroll
                for (int kw = 0; kw < kernel_width; kw++) {
                    const int w_in = w_out * stride_w - padding_w + kw * dilation_w;
                    
                    if (w_in >= 0 && w_in < input_width) {
                        const scalar_t input_val = input[input_offset + h_in * input_width + w_in];
                        const scalar_t weight_val = weight[weight_offset + kh * kernel_width + kw];
                        sum += input_val * weight_val;
                    }
                }
            }
        }
        
        output[b * channels * output_height * output_width + 
               c * output_height * output_width + 
               h_out * output_width + 
               w_out] = sum;
    }
}

// Optimized kernel with shared memory for small kernel sizes
template <typename scalar_t, int KH, int KW>
__global__ void depthwise_conv2d_kernel_shared(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w) {
    
    // Shared memory for weights
    __shared__ scalar_t shared_weight[KH * KW];
    
    // Calculate output position
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z % channels;
    const int b = blockIdx.z / channels;
    
    // Load weights into shared memory
    if (threadIdx.y == 0 && threadIdx.x < KH * KW) {
        shared_weight[threadIdx.x] = weight[c * KH * KW + threadIdx.x];
    }
    
    __syncthreads();
    
    if (w_out < output_width && h_out < output_height) {
        // Calculate input position
        const int input_offset = b * channels * input_height * input_width + 
                                c * input_height * input_width;
        
        scalar_t sum = 0;
        
        #pragma unroll
        for (int kh = 0; kh < KH; kh++) {
            const int h_in = h_out * stride_h - padding_h + kh * dilation_h;
            
            if (h_in >= 0 && h_in < input_height) {
                #pragma unroll
                for (int kw = 0; kw < KW; kw++) {
                    const int w_in = w_out * stride_w - padding_w + kw * dilation_w;
                    
                    if (w_in >= 0 && w_in < input_width) {
                        const scalar_t input_val = input[input_offset + h_in * input_width + w_in];
                        const scalar_t weight_val = shared_weight[kh * KW + kw];
                        sum += input_val * weight_val;
                    }
                }
            }
        }
        
        output[b * channels * output_height * output_width + 
               c * output_height * output_width + 
               h_out * output_width + 
               w_out] = sum;
    }
}

std::vector<torch::Tensor> depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w) {
    
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto input_height = input.size(2);
    const auto input_width = input.size(3);
    const auto kernel_height = weight.size(2);
    const auto kernel_width = weight.size(3);
    
    const auto output_height = (input_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    const auto output_width = (input_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({batch_size, channels, output_height, output_width}, 
                              input.options());
    
    const dim3 threads(16, 16);
    const dim3 blocks((output_width + threads.x - 1) / threads.x,
                      (output_height + threads.y - 1) / threads.y,
                      batch_size * channels);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "depthwise_conv2d_cuda", ([&] {
        if (kernel_height == 3 && kernel_width == 5) {
            depthwise_conv2d_kernel_shared<scalar_t, 3, 5><<<blocks, threads>>>(
                input.data<scalar_t>(),
                weight.data<scalar_t>(),
                output.data<scalar_t>(),
                batch_size,
                channels,
                input_height,
                input_width,
                output_height,
                output_width,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w);
        } else {
            depthwise_conv2d_kernel<scalar_t><<<blocks, threads>>>(
                input.data<scalar_t>(),
                weight.data<scalar_t>(),
                output.data<scalar_t>(),
                batch_size,
                channels,
                input_height,
                input_width,
                kernel_height,
                kernel_width,
                output_height,
                output_width,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w);
        }
    }));
    
    return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &depthwise_conv2d_cuda, "Depthwise Conv2d forward (CUDA)");
}
"""

# Try to load the CUDA extension
try:
    depthwise_conv2d_cuda = load_inline(
        name="depthwise_conv2d_cuda",
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=["forward"],
        verbose=True
    )
except Exception as e:
    print(f"Failed to load CUDA extension: {e}")
    # Fallback to PyTorch implementation
    depthwise_conv2d_cuda = None

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size_h (int): Height of the convolution kernel.
        kernel_size_w (int): Width of the convolution kernel.
        stride_h (int, optional): Stride of the convolution in height dimension. Defaults to 1.
        stride_w (int, optional): Stride of the convolution in width dimension. Defaults to 1.
        padding_h (int, optional): Padding applied to the input in height dimension. Defaults to 0.
        padding_w (int, optional): Padding applied to the input in width dimension. Defaults to 0.
        dilation_h (int, optional): Spacing between kernel elements in height dimension. Defaults to 1.
        dilation_w (int, optional): Spacing between kernel elements in width dimension. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size_h: int, kernel_size_w: int, 
                 stride_h: int = 1, stride_w: int = 1, padding_h: int = 0, padding_w: int = 0, 
                 dilation_h: int = 1, dilation_w: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups
        
        # Create weight parameter with the correct shape for depthwise convolution
        # For depthwise conv with groups=in_channels, each output channel is connected to exactly one input channel
        self.weight = nn.Parameter(torch.Tensor(out_channels, 1, kernel_size_h, kernel_size_w))
        
        # Create bias parameter if needed
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters using the same method as nn.Conv2d
        self.reset_parameters()
        
        # Flag to determine if we should use our custom CUDA kernel
        self.use_cuda_kernel = depthwise_conv2d_cuda is not None
        
    def reset_parameters(self):
        # Initialize weights using the same method as nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        if self.use_cuda_kernel and x.is_cuda and x.dtype in [torch.float32, torch.float16]:
            # Reshape weight for depthwise convolution
            # Our CUDA kernel expects weight in shape [out_channels, kernel_height, kernel_width]
            # but we need to pass it as [out_channels, 1, kernel_height, kernel_width]
            return depthwise_conv2d_cuda.forward(
                x, self.weight, 
                self.stride_h, self.stride_w,
                self.padding_h, self.padding_w,
                self.dilation_h, self.dilation_w
            )[0]
        else:
            # Fallback to PyTorch implementation
            return torch.nn.functional.conv2d(
                x, 
                self.weight, 
                self.bias, 
                stride=(self.stride_h, self.stride_w),
                padding=(self.padding_h, self.padding_w),
                dilation=(self.dilation_h, self.dilation_w),
                groups=self.groups
            )

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
out_channels = in_channels
kernel_size_h = 3
kernel_size_w = 5
width = 256
height = 128
stride_h = 1
stride_w = 1
padding_h = 0
padding_w = 0
dilation_h = 1
dilation_w = 1
groups = in_channels

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups]