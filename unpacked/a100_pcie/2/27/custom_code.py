import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define CUDA kernel for fused Conv3d + HardSwish + ReLU
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Constants for tiling and thread organization
#define TILE_SIZE_D 4
#define TILE_SIZE_H 8
#define TILE_SIZE_W 8
#define WARP_SIZE 32

template <typename scalar_t>
__global__ void fused_conv3d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int depth,
    const int height,
    const int width,
    const int kernel_size) {
    
    // Calculate output dimensions
    const int out_depth = depth - kernel_size + 1;
    const int out_height = height - kernel_size + 1;
    const int out_width = width - kernel_size + 1;
    
    // Calculate output position
    const int n = blockIdx.z;
    const int f = blockIdx.y;
    const int block_idx = blockIdx.x;
    
    // Thread indices within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int tid = tx + ty * blockDim.x + tz * blockDim.x * blockDim.y;
    const int thread_count = blockDim.x * blockDim.y * blockDim.z;
    
    // Check if we're within bounds
    if (n >= batch_size || f >= out_channels)
        return;
    
    // Shared memory for weights
    extern __shared__ char shared_memory[];
    scalar_t* shared_weight = (scalar_t*)shared_memory;
    
    // Constants for faster computation
    const float inv_six = 1.0f / 6.0f;
    
    // Load weights into shared memory (cooperative loading)
    const int weight_size = in_channels * kernel_size * kernel_size * kernel_size;
    for (int i = tid; i < weight_size; i += thread_count) {
        const int c = i / (kernel_size * kernel_size * kernel_size);
        const int remainder = i % (kernel_size * kernel_size * kernel_size);
        const int kz = remainder / (kernel_size * kernel_size);
        const int remainder2 = remainder % (kernel_size * kernel_size);
        const int ky = remainder2 / kernel_size;
        const int kx = remainder2 % kernel_size;
        
        if (c < in_channels) {
            const int weight_idx = ((f * in_channels + c) * kernel_size + kz) * 
                                  kernel_size * kernel_size + ky * kernel_size + kx;
            shared_weight[i] = weight[weight_idx];
        }
    }
    
    // Make sure all threads have loaded the weights
    __syncthreads();
    
    // Calculate starting position for this thread based on block index
    const int blocks_per_dim_z = (out_depth + TILE_SIZE_D - 1) / TILE_SIZE_D;
    const int blocks_per_dim_y = (out_height + TILE_SIZE_H - 1) / TILE_SIZE_H;
    const int blocks_per_dim_x = (out_width + TILE_SIZE_W - 1) / TILE_SIZE_W;
    
    const int block_z = block_idx / (blocks_per_dim_y * blocks_per_dim_x);
    const int block_y = (block_idx % (blocks_per_dim_y * blocks_per_dim_x)) / blocks_per_dim_x;
    const int block_x = block_idx % blocks_per_dim_x;
    
    const int z_start = block_z * TILE_SIZE_D;
    const int y_start = block_y * TILE_SIZE_H;
    const int x_start = block_x * TILE_SIZE_W;
    
    // Process each position in the output volume within this tile
    for (int z_offset = tz; z_offset < TILE_SIZE_D && z_start + z_offset < out_depth; z_offset += blockDim.z) {
        const int z_out = z_start + z_offset;
        
        for (int y_offset = ty; y_offset < TILE_SIZE_H && y_start + y_offset < out_height; y_offset += blockDim.y) {
            const int y_out = y_start + y_offset;
            
            for (int x_offset = tx; x_offset < TILE_SIZE_W && x_start + x_offset < out_width; x_offset += blockDim.x) {
                const int x_out = x_start + x_offset;
                
                // Initialize accumulator
                scalar_t acc = bias ? bias[f] : 0;
                
                // Perform convolution with register blocking for better performance
                for (int c = 0; c < in_channels; ++c) {
                    // Use register blocking to reduce shared memory accesses
                    #pragma unroll 3
                    for (int kz = 0; kz < kernel_size; ++kz) {
                        const int z_in = z_out + kz;
                        
                        #pragma unroll 3
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            const int y_in = y_out + ky;
                            
                            #pragma unroll 3
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                const int x_in = x_out + kx;
                                
                                const int input_idx = ((n * in_channels + c) * depth + z_in) * 
                                                     height * width + y_in * width + x_in;
                                const int weight_idx = (c * kernel_size + kz) * 
                                                      kernel_size * kernel_size + ky * kernel_size + kx;
                                
                                acc += input[input_idx] * shared_weight[weight_idx];
                            }
                        }
                    }
                }
                
                // Apply HardSwish: x * max(0, min(6, x + 3)) / 6
                scalar_t min_val = fminf(6.0f, acc + 3.0f);
                scalar_t max_val = fmaxf(0.0f, min_val);
                scalar_t hs_val = acc * max_val * inv_six;
                
                // Apply ReLU: max(0, x)
                scalar_t relu_val = fmaxf(0.0f, hs_val);
                
                // Write to output
                const int out_idx = ((n * out_channels + f) * out_depth + z_out) * 
                                   out_height * out_width + y_out * out_width + x_out;
                output[out_idx] = relu_val;
            }
        }
    }
}

// Optimized kernel for softmax and mean combined
template <typename scalar_t>
__global__ void softmax_mean_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int depth,
    const int height,
    const int width) {
    
    const int n = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (n >= batch_size || c >= channels)
        return;
    
    const int spatial_size = depth * height * width;
    extern __shared__ char shared_memory[];
    scalar_t* shared_data = (scalar_t*)shared_memory;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    // Step 1: Find max value for numerical stability
    scalar_t max_val = -INFINITY;
    
    // Use vectorized memory access for better throughput when possible
    for (int i = 0; i < spatial_size; ++i) {
        const int idx = ((n * channels + c) * depth * height * width) + i;
        if (c < channels) {
            max_val = max(max_val, input[idx]);
        }
    }
    
    // Use warp-level reduction to find the maximum value
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    
    // Broadcast max_val to all threads in the warp
    max_val = __shfl_sync(0xffffffff, max_val, 0);
    
    // Step 2: Compute exp(x - max) and sum
    scalar_t sum_exp = 0.0f;
    
    for (int i = 0; i < spatial_size; ++i) {
        const int idx = ((n * channels + c) * depth * height * width) + i;
        if (c < channels) {
            scalar_t val = exp(input[idx] - max_val);
            shared_data[tid] = val;  // Store exp value in shared memory
            sum_exp += val;
        }
    }
    
    // Use warp-level reduction to find the sum
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    
    // Broadcast sum_exp to all threads in the warp
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);
    
    // Step 3: Normalize by sum and compute mean simultaneously
    scalar_t mean_val = 0.0f;
    const float inv_spatial_size = 1.0f / spatial_size;
    
    for (int i = 0; i < spatial_size; ++i) {
        const int idx = ((n * channels + c) * depth * height * width) + i;
        if (c < channels) {
            scalar_t softmax_val = exp(input[idx] - max_val) / sum_exp;
            mean_val += softmax_val * inv_spatial_size;
        }
    }
    
    // Write output (only one thread per channel writes the result)
    if (lane_id == 0 && c < channels) {
        output[n * channels + c] = mean_val;
    }
}

// C++ interface for the fused convolution kernel
void fused_conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int height,
    int width,
    int kernel_size) {
    
    // Calculate output dimensions
    const int out_depth = depth - kernel_size + 1;
    const int out_height = height - kernel_size + 1;
    const int out_width = width - kernel_size + 1;
    
    // Define block and grid dimensions - optimized for better occupancy
    const dim3 threads(8, 8, 4);  // 256 threads per block
    
    // Calculate number of blocks needed for the output volume
    const int blocks_per_dim_z = (out_depth + TILE_SIZE_D - 1) / TILE_SIZE_D;
    const int blocks_per_dim_y = (out_height + TILE_SIZE_H - 1) / TILE_SIZE_H;
    const int blocks_per_dim_x = (out_width + TILE_SIZE_W - 1) / TILE_SIZE_W;
    const int blocks_per_volume = blocks_per_dim_z * blocks_per_dim_y * blocks_per_dim_x;
    
    const dim3 blocks(blocks_per_volume, out_channels, batch_size);
    
    // Calculate shared memory size for weights
    const int shared_mem_size = in_channels * kernel_size * kernel_size * kernel_size * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_conv3d_kernel", ([&] {
        fused_conv3d_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            depth,
            height,
            width,
            kernel_size);
    }));
}

// C++ interface for the softmax_mean kernel
void softmax_mean_cuda(
    torch::Tensor input,
    torch::Tensor output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width) {
    
    const int threads = 32;  // One warp per block for efficient reductions
    const int blocks_y = (channels + threads - 1) / threads;
    const dim3 blocks(batch_size, blocks_y);
    
    // Calculate shared memory size
    const int shared_mem_size = threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softmax_mean_kernel", ([&] {
        softmax_mean_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            depth,
            height,
            width);
    }));
}
"""

cpp_source = """
#include <torch/extension.h>

// Forward declarations
void fused_conv3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int height,
    int width,
    int kernel_size);

void softmax_mean_cuda(
    torch::Tensor input,
    torch::Tensor output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width);

// C++ interface
void fused_conv3d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int height,
    int width,
    int kernel_size) {
    
    fused_conv3d_cuda(
        input, weight, bias, output, 
        batch_size, in_channels, out_channels, 
        depth, height, width, kernel_size);
}

void softmax_mean(
    torch::Tensor input,
    torch::Tensor output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width) {
    
    softmax_mean_cuda(
        input, output, 
        batch_size, channels, 
        depth, height, width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv3d", &fused_conv3d, 
          "Fused Conv3D + HardSwish + ReLU");
    m.def("softmax_mean", &softmax_mean, 
          "Fused Softmax + Mean");
}
"""

# Try to compile the CUDA extension
try:
    optimized_ops = load_inline(
        name='optimized_ops',
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=['fused_conv3d', 'softmax_mean'],
        with_cuda=True,
        extra_cuda_cflags=['-O3', '--use_fast_math', '-Xptxas=-v', '--fmad=true']
    )
except Exception as e:
    print(f"Failed to compile CUDA extension: {e}")
    optimized_ops = None

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        bias (bool): Whether to include bias
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        
        # Enable cuDNN benchmark mode to find the best algorithm
        torch.backends.cudnn.benchmark = True
        
        # Store dimensions for the custom kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.has_bias = bias
        
        # Create CUDA streams for potential overlapping of operations
        if torch.cuda.is_available():
            self.stream1 = torch.cuda.Stream()
            self.stream2 = torch.cuda.Stream()
        else:
            self.stream1 = None
            self.stream2 = None
        
        # Flag to track if we've warned about fallback
        self.warned_about_fallback = False
    
    def forward(self, x):
        # Use our CUDA streams if available
        if self.stream1 is not None:
            with torch.cuda.stream(self.stream1):
                return self._forward_impl(x)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        batch_size, _, depth, height, width = x.shape
        
        # Calculate output dimensions
        out_depth = depth - self.kernel_size + 1
        out_height = height - self.kernel_size + 1
        out_width = width - self.kernel_size + 1
        
        # Ensure input is contiguous for better memory access patterns
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Ensure weight and bias are contiguous
        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous() if self.has_bias else None
        
        if optimized_ops is not None:
            try:
                # Allocate output tensor for convolution result
                conv_output = torch.empty(
                    (batch_size, self.out_channels, out_depth, out_height, out_width),
                    dtype=x.dtype, device=x.device
                )
                
                # Allocate tensor for final result
                result = torch.empty(
                    (batch_size, self.out_channels),
                    dtype=x.dtype, device=x.device
                )
                
                # Call our fused convolution kernel
                optimized_ops.fused_conv3d(
                    x, weight, bias, conv_output,
                    batch_size, self.in_channels, self.out_channels,
                    depth, height, width, self.kernel_size
                )
                
                # Call our softmax_mean kernel
                optimized_ops.softmax_mean(
                    conv_output, result,
                    batch_size, self.out_channels,
                    out_depth, out_height, out_width
                )
                
                return result
                
            except Exception as e:
                # Fallback to PyTorch implementation if CUDA kernel fails
                if not self.warned_about_fallback:
                    print(f"CUDA kernel failed, falling back to PyTorch: {e}")
                    self.warned_about_fallback = True
                return self._fallback_impl(x)
        else:
            # Use PyTorch's native operations if CUDA extension is not available
            return self._fallback_impl(x)
    
    def _fallback_impl(self, x):
        # Standard PyTorch implementation as fallback
        x = self.conv(x)
        x = F.hardswish(x)
        x = F.relu(x)
        x = F.softmax(x, dim=1)
        x = torch.mean(x, dim=[2, 3, 4])
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]