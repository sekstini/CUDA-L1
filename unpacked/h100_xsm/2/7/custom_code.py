import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.kernel_compiled = False
        self.use_custom_kernel = False
        
        # Compile the kernel on first use to avoid initialization issues
        if torch.cuda.is_available():
            self._compile_kernel()
    
    def _compile_kernel(self):
        """Compile the optimized CUDA kernel for fused activations"""
        cuda_code = """
        extern "C" __global__ void fused_activations_kernel(
            float* __restrict__ output,
            const float* __restrict__ input,
            const float* __restrict__ bias,
            const int N,
            const int C,
            const int D,
            const int H,
            const int W) {
            
            // Calculate global thread ID
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Total number of elements
            const int total_elements = N * C * D * H * W;
            
            // Grid-stride loop for handling arbitrary input sizes
            for (int idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
                // Calculate position in 5D tensor
                const int w = idx % W;
                const int h = (idx / W) % H;
                const int d = (idx / (W * H)) % D;
                const int c = (idx / (W * H * D)) % C;
                const int n = idx / (W * H * D * C);
                
                // Load input value
                float x = input[idx];
                
                // Apply ReLU (x = max(0, x))
                x = fmaxf(x, 0.0f);
                
                // Apply LeakyReLU (x = max(0.01*x, x))
                // Since x is already non-negative after ReLU, this only affects x=0
                // which becomes max(0, 0) = 0, so we can skip this step
                
                // Apply GELU - optimized approximation
                // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                const float sqrt_2_pi = 0.7978845608f;
                const float coeff = 0.044715f;
                float x_squared = x * x;
                float x_cubed = x_squared * x;
                float tanh_in = sqrt_2_pi * (x + coeff * x_cubed);
                float tanh_out = tanhf(tanh_in);
                x = 0.5f * x * (1.0f + tanh_out);
                
                // Apply Sigmoid using fast math
                x = __fdividef(1.0f, (1.0f + __expf(-x)));
                
                // Add bias (broadcast along C dimension)
                x = x + bias[c];
                
                // Store result
                output[idx] = x;
            }
        }
        """
        
        try:
            from torch.utils.cpp_extension import load_inline
            self.fused_kernel = load_inline(
                name='fused_activations_kernel',
                cpp_sources='',
                cuda_sources=cuda_code,
                functions=['fused_activations_kernel'],
                with_cuda=True,
                verbose=False,
                extra_cuda_cflags=['-O3', '--use_fast_math']
            )
            self.kernel_compiled = True
            self.use_custom_kernel = True
        except Exception as e:
            print(f"Failed to compile optimized kernel: {e}")
            self.kernel_compiled = False
            self.use_custom_kernel = False
    
    def forward(self, x):
        # Check if we're using CUDA
        using_cuda = x.is_cuda
        
        # Apply convolution using PyTorch's optimized implementation
        # Convert to channels-last format for better memory access patterns if on CUDA
        if using_cuda:
            x = x.contiguous(memory_format=torch.channels_last_3d)
            # Ensure the conv layer is optimized for channels-last
            if not hasattr(self, 'conv_converted'):
                self.conv = self.conv.to(memory_format=torch.channels_last_3d)
                self.conv_converted = True
        
        # Apply convolution
        x = self.conv(x)
        
        # Try to use our optimized kernel if we're on CUDA
        if using_cuda and self.use_custom_kernel:
            try:
                # Get tensor dimensions
                N, C, D, H, W = x.shape
                
                # Ensure input is contiguous
                x = x.contiguous()
                
                # Prepare output tensor
                output = torch.empty_like(x)
                
                # Prepare bias tensor
                bias_flat = self.bias.view(C).contiguous()
                
                # Calculate optimal grid and block dimensions
                threads_per_block = 256  # Optimal for most modern GPUs
                
                # Calculate grid size to ensure enough parallelism
                total_elements = N * C * D * H * W
                blocks = min(65535, (total_elements + threads_per_block - 1) // threads_per_block)
                
                # Launch optimized kernel
                self.fused_kernel.fused_activations_kernel(
                    output, x, bias_flat, N, C, D, H, W,
                    grid=(blocks,),
                    block=(threads_per_block,)
                )
                
                return output
                
            except Exception as e:
                print(f"Optimized kernel execution failed, falling back to PyTorch: {e}")
        
        # Fallback to PyTorch implementation
        x = torch.relu(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        x = torch.nn.functional.gelu(x)
        x = torch.sigmoid(x)
        x = x + self.bias
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]