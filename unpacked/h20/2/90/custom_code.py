import torch
import torch.nn as nn
import torch.utils.cpp_extension
import os
import math

# Enable cuDNN benchmark mode to find the best algorithm
torch.backends.cudnn.benchmark = True

# Create a directory for our CUDA code
os.makedirs('cuda_code', exist_ok=True)

# Write CUDA kernel code
with open('cuda_code/fused_ops_kernel.cu', 'w') as f:
    f.write('''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel for fused operations: LeakyReLU -> Add -> Clamp -> GELU
__global__ void fused_ops_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ sum_tensor,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    float negative_slope) {
    
    const int total_elements = batch_size * channels * depth * height * width;
    const int dhw = depth * height * width;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_elements; 
         idx += blockDim.x * gridDim.x) {
        
        // Calculate channel index for broadcasting
        const int c = (idx / dhw) % channels;
        
        // Get input value
        const float x = input[idx];
        
        // Apply LeakyReLU
        float result = x > 0 ? x : x * negative_slope;
        
        // Add sum_tensor (broadcasting)
        result += sum_tensor[c];
        
        // Apply clamp
        result = fminf(1.0f, fmaxf(-1.0f, result));
        
        // Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float sqrt_2_pi_inv = 0.7978845608028654f;  // sqrt(2/pi)
        const float coeff = 0.044715f;
        const float x_cubed = result * result * result;
        const float inner = sqrt_2_pi_inv * (result + coeff * x_cubed);
        result = 0.5f * result * (1.0f + tanhf(inner));
        
        // Store result
        output[idx] = result;
    }
}

torch::Tensor fused_ops_forward_cuda(
    torch::Tensor input,
    torch::Tensor sum_tensor,
    float negative_slope) {
    
    auto output = torch::zeros_like(input);
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);
    
    // Optimize thread and block configuration
    const int threads = 256;
    const int blocks = min(65535, (batch_size * channels * depth * height * width + threads - 1) / threads);
    
    fused_ops_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        sum_tensor.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width,
        negative_slope
    );
    
    return output;
}
''')

# Write C++ interface
with open('cuda_code/fused_ops.cpp', 'w') as f:
    f.write('''
#include <torch/extension.h>

// Forward declaration of CUDA function
torch::Tensor fused_ops_forward_cuda(
    torch::Tensor input,
    torch::Tensor sum_tensor,
    float negative_slope);

// C++ interface
torch::Tensor fused_ops_forward(
    torch::Tensor input,
    torch::Tensor sum_tensor,
    float negative_slope) {
    
    return fused_ops_forward_cuda(input, sum_tensor, negative_slope);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_ops_forward, "Fused Ops Forward");
}
''')

# Try to compile the extension
try:
    fused_ops_cuda = torch.utils.cpp_extension.load(
        name="fused_ops_cuda",
        sources=["cuda_code/fused_ops.cpp", "cuda_code/fused_ops_kernel.cu"],
        verbose=True,
        extra_cuda_cflags=["--use_fast_math", "-O3"]  # Enable fast math and high optimization
    )
    has_cuda_extension = True
except Exception as e:
    print(f"Failed to compile CUDA extension: {e}")
    print("Falling back to PyTorch implementation.")
    has_cuda_extension = False

class FusedElementwiseOps(torch.autograd.Function):
    """
    Custom autograd function for fused element-wise operations:
    - LeakyReLU
    - Addition with sum_tensor
    - Clamp
    - GELU
    """
    @staticmethod
    def forward(ctx, x, sum_tensor, negative_slope=0.2):
        # Save for backward
        ctx.save_for_backward(x, sum_tensor)
        ctx.negative_slope = negative_slope
        
        # Use custom CUDA kernel if available
        if has_cuda_extension and x.is_cuda and sum_tensor.is_cuda:
            try:
                return fused_ops_cuda.forward(x, sum_tensor, negative_slope)
            except Exception as e:
                print(f"Error in CUDA kernel: {e}")
                print("Falling back to PyTorch implementation.")
        
        # Fallback to PyTorch implementation
        leaky_relu = torch.nn.functional.leaky_relu(x, negative_slope)
        added = leaky_relu + sum_tensor
        clamped = torch.clamp(added, min=-1.0, max=1.0)
        result = torch.nn.functional.gelu(clamped)
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, sum_tensor = ctx.saved_tensors
        negative_slope = ctx.negative_slope
        
        # Compute intermediate results for backward pass
        leaky_relu = torch.nn.functional.leaky_relu(x, negative_slope)
        added = leaky_relu + sum_tensor
        clamped = torch.clamp(added, min=-1.0, max=1.0)
        
        # GELU gradient
        gelu_grad = 0.5 * (1.0 + torch.erf(clamped / math.sqrt(2.0)))
        gelu_grad = gelu_grad + (clamped * torch.exp(-0.5 * clamped**2) / math.sqrt(2.0 * math.pi))
        
        # Clamp gradient
        clamp_grad = torch.ones_like(added)
        clamp_grad[added < -1.0] = 0
        clamp_grad[added > 1.0] = 0
        
        # LeakyReLU gradient
        leaky_grad = torch.ones_like(x)
        leaky_grad[x < 0] = negative_slope
        
        # Combine gradients
        grad_x = grad_output * gelu_grad * clamp_grad * leaky_grad
        grad_sum = grad_output * gelu_grad * clamp_grad
        
        # Sum over appropriate dimensions for grad_sum
        grad_sum = grad_sum.sum(dim=(0, 2, 3, 4), keepdim=True)
        
        return grad_x, grad_sum, None

class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.fused_ops = FusedElementwiseOps.apply
        
        # Convert weights to channels_last_3d format during initialization
        self.conv.weight.data = self.conv.weight.data.to(memory_format=torch.channels_last_3d)
        if self.conv.bias is not None:
            self.conv.bias.data = self.conv.bias.data.contiguous()

    def forward(self, x):
        # Convert input to channels_last_3d format for better memory access patterns
        x = x.to(memory_format=torch.channels_last_3d)
        
        # Apply convolution
        x = self.conv(x)
        
        # Apply fused element-wise operations
        x = self.fused_ops(x, self.sum_tensor, 0.2)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
sum_tensor_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]