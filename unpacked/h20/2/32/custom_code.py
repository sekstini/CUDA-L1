import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Ultra-optimized single-kernel fused CUDA implementation
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 32
#define TILE_SIZE 34  // 32 + 2 (kernel_size - 1)

__global__ void ultra_optimized_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int out_channels
) {
    extern __shared__ float shared_input[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int batch_idx = blockIdx.z;
    
    int out_x = bx * BLOCK_SIZE + tx;
    int out_y = by * BLOCK_SIZE + ty;
    
    // Early exit for out-of-bounds threads
    bool valid_thread = (out_x < output_width && out_y < output_height);
    
    float min_val = INFINITY;
    
    // Process each input channel
    for (int ic = 0; ic < in_channels; ic++) {
        // Load input tile into shared memory with vectorized access where possible
        int input_base = batch_idx * in_channels * input_height * input_width + 
                        ic * input_height * input_width;
        
        // Each thread loads multiple elements to fill the shared memory tile
        int tid = ty * BLOCK_SIZE + tx;
        int total_threads = BLOCK_SIZE * BLOCK_SIZE;
        
        for (int i = tid; i < TILE_SIZE * TILE_SIZE; i += total_threads) {
            int tile_y = i / TILE_SIZE;
            int tile_x = i % TILE_SIZE;
            int global_y = by * BLOCK_SIZE + tile_y;
            int global_x = bx * BLOCK_SIZE + tile_x;
            
            if (global_y < input_height && global_x < input_width) {
                shared_input[i] = input[input_base + global_y * input_width + global_x];
            } else {
                shared_input[i] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute convolution for all output channels if thread is valid
        if (valid_thread) {
            for (int oc = 0; oc < out_channels; oc++) {
                float conv_sum = bias[oc];  // Pre-scaled bias
                
                // Unrolled 3x3 convolution using shared memory
                int shared_base = ty * TILE_SIZE + tx;
                int weight_base = oc * in_channels * 9 + ic * 9;
                
                conv_sum += shared_input[shared_base] * weight[weight_base];
                conv_sum += shared_input[shared_base + 1] * weight[weight_base + 1];
                conv_sum += shared_input[shared_base + 2] * weight[weight_base + 2];
                conv_sum += shared_input[shared_base + TILE_SIZE] * weight[weight_base + 3];
                conv_sum += shared_input[shared_base + TILE_SIZE + 1] * weight[weight_base + 4];
                conv_sum += shared_input[shared_base + TILE_SIZE + 2] * weight[weight_base + 5];
                conv_sum += shared_input[shared_base + 2 * TILE_SIZE] * weight[weight_base + 6];
                conv_sum += shared_input[shared_base + 2 * TILE_SIZE + 1] * weight[weight_base + 7];
                conv_sum += shared_input[shared_base + 2 * TILE_SIZE + 2] * weight[weight_base + 8];
                
                // Update running minimum
                min_val = fminf(min_val, conv_sum);
            }
        }
        
        __syncthreads();
    }
    
    // Write final result
    if (valid_thread) {
        int output_idx = batch_idx * output_height * output_width + 
                        out_y * output_width + out_x;
        output[output_idx] = min_val;
    }
}

torch::Tensor ultra_optimized_fused_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_height = input.size(2);
    auto input_width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto output_height = input_height - 2; // 3x3 kernel, no padding
    auto output_width = input_width - 2;
    
    // Final output tensor
    auto output = torch::zeros({batch_size, 1, output_height, output_width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    
    // Launch configuration optimized for our problem size
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
              batch_size);
    
    // Shared memory for input tile
    int shared_mem_size = TILE_SIZE * TILE_SIZE * sizeof(float);
    
    ultra_optimized_fused_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        out_channels
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor ultra_optimized_fused_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);

torch::Tensor ultra_optimized_fused(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    return ultra_optimized_fused_cuda(input, weight, bias);
}
"""

# Compile the CUDA extension
try:
    ultra_ops = load_inline(
        name='ultra_optimized_fused',
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=['ultra_optimized_fused'],
        verbose=False,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math', '-Xptxas=-v', '--maxrregcount=32']
    )
    CUDA_AVAILABLE = True
except Exception as e:
    CUDA_AVAILABLE = False
    print(f"CUDA compilation failed, falling back to PyTorch: {e}")

class ModelNew(nn.Module):
    """
    Model that performs a convolution, scales the output, and then applies a minimum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        
        # Create temporary Conv2d to get properly initialized weights
        temp_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=True)
        
        # Pre-scale weights and bias to eliminate runtime scaling
        with torch.no_grad():
            scaled_weight = temp_conv.weight.data * scale_factor
            scaled_bias = temp_conv.bias.data * scale_factor
            
            # Store as contiguous buffers for optimal memory access
            self.register_buffer('weight', scaled_weight.contiguous())
            self.register_buffer('bias', scaled_bias.contiguous())
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        if CUDA_AVAILABLE and x.is_cuda:
            # Use ultra-optimized fused CUDA kernel
            return ultra_ops.ultra_optimized_fused(x, self.weight, self.bias)
        else:
            # Fallback to optimized PyTorch implementation
            conv_out = F.conv2d(x, self.weight, self.bias)
            return torch.amin(conv_out, dim=1, keepdim=True)

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