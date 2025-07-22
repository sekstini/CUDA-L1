import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define CUDA kernel for fused convolution, scaling, and minimum reduction
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void fused_conv_scale_min_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int out_height,
    const int out_width,
    const float scale_factor) {
    
    // Calculate output position
    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for weights
    extern __shared__ char shared_mem[];
    scalar_t* shared_weights = reinterpret_cast<scalar_t*>(shared_mem);
    
    // Only proceed if this thread corresponds to a valid output pixel
    if (h < out_height && w < out_width) {
        // Thread-local minimum value (initialize with a large value)
        scalar_t min_val = 1e10f;
        
        // Process all output channels
        for (int oc = 0; oc < out_channels; ++oc) {
            scalar_t conv_result = 0;
            
            // Apply convolution
            for (int ic = 0; ic < in_channels; ++ic) {
                // Load weights into shared memory
                const int weights_per_thread = (in_channels * kernel_size * kernel_size + blockDim.x * blockDim.y - 1) / (blockDim.x * blockDim.y);
                const int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
                
                for (int i = 0; i < weights_per_thread; ++i) {
                    const int idx = thread_idx * weights_per_thread + i;
                    if (idx < kernel_size * kernel_size) {
                        const int kh = idx / kernel_size;
                        const int kw = idx % kernel_size;
                        if (kh < kernel_size && kw < kernel_size) {
                            const int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                            shared_weights[idx] = weight[weight_idx];
                        }
                    }
                }
                __syncthreads();
                
                // Perform convolution using shared weights
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        const int ih = h + kh;
                        const int iw = w + kw;
                        
                        if (ih < in_height && iw < in_width) {
                            const int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                            const int weight_idx = kh * kernel_size + kw;
                            
                            conv_result += input[input_idx] * shared_weights[weight_idx];
                        }
                    }
                }
                __syncthreads();
            }
            
            // Add bias and apply scaling
            if (bias != nullptr) {
                conv_result += bias[oc];
            }
            conv_result *= scale_factor;
            
            // Update minimum value
            min_val = min(min_val, conv_result);
        }
        
        // Write result to output
        const int output_idx = (b * out_height + h) * out_width + w;
        output[output_idx] = min_val;
    }
}

std::vector<torch::Tensor> fused_conv_scale_min_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor) {
    
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto in_height = input.size(2);
    const auto in_width = input.size(3);
    const auto out_channels = weight.size(0);
    const auto kernel_size = weight.size(2);
    
    const auto out_height = in_height - kernel_size + 1;
    const auto out_width = in_width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, 1, out_height, out_width}, 
                              input.options());
    
    // Optimized thread block configuration
    const dim3 threads(32, 8);
    const dim3 blocks((out_width + threads.x - 1) / threads.x,
                     (out_height + threads.y - 1) / threads.y,
                     batch_size);
    
    // Calculate shared memory size
    const int shared_mem_size = kernel_size * kernel_size * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_conv_scale_min_kernel", ([&] {
        fused_conv_scale_min_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            in_height,
            in_width,
            kernel_size,
            out_height,
            out_width,
            scale_factor);
    }));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_scale_min", &fused_conv_scale_min_cuda, "Fused convolution, scaling, and minimum reduction (CUDA)");
}
"""

# Try to load the CUDA extension
try:
    fused_ops = load_inline(
        name="optimized_fused_ops",
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=["fused_conv_scale_min"],
        verbose=False,
        extra_cuda_cflags=['-O3', '--use_fast_math']
    )
except Exception as e:
    print(f"CUDA extension compilation failed: {e}")
    fused_ops = None

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolution kernel
        scale_factor (float): Scaling factor to apply
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        
        # Pre-scale weights and bias for fallback optimization
        with torch.no_grad():
            self.register_buffer('scaled_weight', self.conv.weight.clone() * scale_factor)
            if self.conv.bias is not None:
                self.register_buffer('scaled_bias', self.conv.bias.clone() * scale_factor)
            else:
                self.register_buffer('scaled_bias', None)
        
        self.use_custom_kernel = fused_ops is not None
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, height-kernel_size+1, width-kernel_size+1)
        """
        # Ensure optimal memory layout
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Try custom CUDA kernel first
        if self.use_custom_kernel and x.is_cuda:
            try:
                out_height = x.size(2) - self.conv.weight.size(2) + 1
                out_width = x.size(3) - self.conv.weight.size(2) + 1
                result = fused_ops.fused_conv_scale_min(
                    x, 
                    self.conv.weight,
                    self.conv.bias if self.conv.bias is not None else torch.tensor([]).to(x.device),
                    self.scale_factor
                )[0]
                return result.view(x.size(0), 1, out_height, out_width)
            except Exception as e:
                print(f"Custom kernel failed, using fallback: {e}")
                self.use_custom_kernel = False
        
        # Optimized fallback using pre-scaled weights
        x = F.conv2d(x, self.scaled_weight, self.scaled_bias)
        return torch.amin(x, dim=1, keepdim=True)  # Using amin instead of min for better performance

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]