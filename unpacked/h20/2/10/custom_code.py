import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for fused ConvTranspose2d + MaxPool + Hardtanh + Mean + Tanh
template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void fused_convtranspose_maxpool_hardtanh_mean_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width,
    int kernel_size, int stride, int padding,
    int maxpool_kernel_size, int maxpool_stride,
    float hardtanh_min, float hardtanh_max) {
    
    // Each block processes one batch item and one output channel
    int b = blockIdx.x;
    int oc = blockIdx.y;
    
    // Check bounds
    if (b >= batch_size || oc >= out_channels)
        return;
    
    // Calculate dimensions
    int conv_out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    int conv_out_width = (in_width - 1) * stride - 2 * padding + kernel_size;
    int pool_out_height = (conv_out_height - maxpool_kernel_size) / maxpool_stride + 1;
    int pool_out_width = (conv_out_width - maxpool_kernel_size) / maxpool_stride + 1;
    
    // Shared memory for weights and partial sums
    extern __shared__ float shared_mem[];
    float* weights_shared = shared_mem;
    float* partial_sums = weights_shared + TILE_SIZE * kernel_size * kernel_size;
    
    // Initialize partial sum for this thread
    if (threadIdx.x < BLOCK_SIZE) {
        partial_sums[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // Load bias
    float bias_val = bias[oc];
    
    // Each thread processes multiple pool output positions
    float thread_sum = 0.0f;
    int valid_count = 0;
    
    // Process pool output positions in chunks
    for (int p_idx = threadIdx.x; p_idx < pool_out_height * pool_out_width; p_idx += blockDim.x) {
        int ph = p_idx / pool_out_width;
        int pw = p_idx % pool_out_width;
        
        // Calculate corresponding conv output region for this pool window
        int ch_start = ph * maxpool_stride;
        int cw_start = pw * maxpool_stride;
        
        // Apply max pooling
        float max_val = -INFINITY;
        
        for (int kh = 0; kh < maxpool_kernel_size; kh++) {
            int ch = ch_start + kh;
            if (ch >= conv_out_height) continue;
            
            for (int kw = 0; kw < maxpool_kernel_size; kw++) {
                int cw = cw_start + kw;
                if (cw >= conv_out_width) continue;
                
                // Compute convolution output for this position
                float conv_val = bias_val;
                
                // Process input channels in tiles to maximize data reuse
                for (int ic_tile = 0; ic_tile < in_channels; ic_tile += TILE_SIZE) {
                    int ic_end = min(ic_tile + TILE_SIZE, in_channels);
                    
                    // Collaboratively load weights for this output channel and input channel tile
                    for (int k_idx = threadIdx.x; k_idx < (ic_end - ic_tile) * kernel_size * kernel_size; k_idx += blockDim.x) {
                        int ic_offset = k_idx / (kernel_size * kernel_size);
                        int k_remainder = k_idx % (kernel_size * kernel_size);
                        int kh_idx = k_remainder / kernel_size;
                        int kw_idx = k_remainder % kernel_size;
                        
                        int ic = ic_tile + ic_offset;
                        
                        weights_shared[ic_offset * kernel_size * kernel_size + kh_idx * kernel_size + kw_idx] = 
                            weight[ic * out_channels * kernel_size * kernel_size + 
                                  oc * kernel_size * kernel_size + 
                                  kh_idx * kernel_size + 
                                  kw_idx];
                    }
                    __syncthreads();
                    
                    // Calculate corresponding input region for this conv output position
                    int ih_start = max(0, (ch + padding - kernel_size + 1 + stride - 1) / stride);
                    int iw_start = max(0, (cw + padding - kernel_size + 1 + stride - 1) / stride);
                    
                    int ih_end = min((ch + padding) / stride + 1, in_height);
                    int iw_end = min((cw + padding) / stride + 1, in_width);
                    
                    // Process input region
                    for (int ih = ih_start; ih < ih_end; ih++) {
                        for (int iw = iw_start; iw < iw_end; iw++) {
                            // Calculate kernel position
                            int kh_conv = ch + padding - ih * stride;
                            int kw_conv = cw + padding - iw * stride;
                            
                            // Check if kernel position is valid
                            if (kh_conv >= 0 && kh_conv < kernel_size && kw_conv >= 0 && kw_conv < kernel_size) {
                                // Process input channels in this tile
                                #pragma unroll 4
                                for (int ic_offset = 0; ic_offset < (ic_end - ic_tile); ic_offset++) {
                                    int ic = ic_tile + ic_offset;
                                    
                                    float input_val = input[b * in_channels * in_height * in_width + 
                                                         ic * in_height * in_width + 
                                                         ih * in_width + 
                                                         iw];
                                    
                                    float weight_val = weights_shared[ic_offset * kernel_size * kernel_size + 
                                                                   kh_conv * kernel_size + 
                                                                   kw_conv];
                                    
                                    conv_val += input_val * weight_val;
                                }
                            }
                        }
                    }
                    __syncthreads();
                }
                
                // Early hardtanh bounds checking
                if (conv_val > hardtanh_max) {
                    conv_val = hardtanh_max;
                } else if (conv_val < hardtanh_min) {
                    conv_val = hardtanh_min;
                }
                
                // Update max value
                max_val = max(max_val, conv_val);
            }
        }
        
        // Add to thread's sum (max_val is already bounded by hardtanh)
        thread_sum += max_val;
        valid_count++;
    }
    
    // Store thread's sum in shared memory
    partial_sums[threadIdx.x] = thread_sum;
    __syncthreads();
    
    // Parallel reduction to compute sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Final mean and tanh
    if (threadIdx.x == 0) {
        float mean = partial_sums[0] / (pool_out_height * pool_out_width);
        output[b * out_channels + oc] = tanh(mean);
    }
}

// Optimized version with warp-level primitives
template <int BLOCK_SIZE, int TILE_SIZE, int WARP_SIZE=32>
__global__ void warp_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_height, int in_width,
    int kernel_size, int stride, int padding,
    int maxpool_kernel_size, int maxpool_stride,
    float hardtanh_min, float hardtanh_max) {
    
    // Each block processes one batch item and one output channel
    int b = blockIdx.x;
    int oc = blockIdx.y;
    
    // Check bounds
    if (b >= batch_size || oc >= out_channels)
        return;
    
    // Calculate dimensions
    int conv_out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    int conv_out_width = (in_width - 1) * stride - 2 * padding + kernel_size;
    int pool_out_height = (conv_out_height - maxpool_kernel_size) / maxpool_stride + 1;
    int pool_out_width = (conv_out_width - maxpool_kernel_size) / maxpool_stride + 1;
    
    // Shared memory for weights and partial sums
    extern __shared__ float shared_mem[];
    float* weights_shared = shared_mem;
    float* partial_sums = weights_shared + TILE_SIZE * kernel_size * kernel_size;
    
    // Get warp and lane indices
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    
    // Load bias
    float bias_val = bias[oc];
    
    // Each thread processes multiple pool output positions
    float thread_sum = 0.0f;
    
    // Process pool output positions by warp
    for (int p_idx_base = warp_id; p_idx_base < pool_out_height * pool_out_width; p_idx_base += num_warps) {
        int ph = p_idx_base / pool_out_width;
        int pw = p_idx_base % pool_out_width;
        
        // Calculate corresponding conv output region for this pool window
        int ch_start = ph * maxpool_stride;
        int cw_start = pw * maxpool_stride;
        
        // Each lane processes different positions in the maxpool window
        float lane_max = -INFINITY;
        
        for (int mp_offset = lane_id; mp_offset < maxpool_kernel_size * maxpool_kernel_size; mp_offset += WARP_SIZE) {
            int kh = mp_offset / maxpool_kernel_size;
            int kw = mp_offset % maxpool_kernel_size;
            
            int ch = ch_start + kh;
            int cw = cw_start + kw;
            
            // Check bounds
            if (ch < conv_out_height && cw < conv_out_width) {
                // Compute convolution output for this position
                float conv_val = bias_val;
                
                // Process input channels in tiles to maximize data reuse
                for (int ic_tile = 0; ic_tile < in_channels; ic_tile += TILE_SIZE) {
                    int ic_end = min(ic_tile + TILE_SIZE, in_channels);
                    
                    // Collaboratively load weights for this output channel and input channel tile
                    for (int k_idx = threadIdx.x; k_idx < (ic_end - ic_tile) * kernel_size * kernel_size; k_idx += blockDim.x) {
                        int ic_offset = k_idx / (kernel_size * kernel_size);
                        int k_remainder = k_idx % (kernel_size * kernel_size);
                        int kh_idx = k_remainder / kernel_size;
                        int kw_idx = k_remainder % kernel_size;
                        
                        int ic = ic_tile + ic_offset;
                        
                        weights_shared[ic_offset * kernel_size * kernel_size + kh_idx * kernel_size + kw_idx] = 
                            weight[ic * out_channels * kernel_size * kernel_size + 
                                  oc * kernel_size * kernel_size + 
                                  kh_idx * kernel_size + 
                                  kw_idx];
                    }
                    __syncthreads();
                    
                    // Calculate corresponding input region for this conv output position
                    int ih_start = max(0, (ch + padding - kernel_size + 1 + stride - 1) / stride);
                    int iw_start = max(0, (cw + padding - kernel_size + 1 + stride - 1) / stride);
                    
                    int ih_end = min((ch + padding) / stride + 1, in_height);
                    int iw_end = min((cw + padding) / stride + 1, in_width);
                    
                    // Process input region
                    for (int ih = ih_start; ih < ih_end; ih++) {
                        for (int iw = iw_start; iw < iw_end; iw++) {
                            // Calculate kernel position
                            int kh_conv = ch + padding - ih * stride;
                            int kw_conv = cw + padding - iw * stride;
                            
                            // Check if kernel position is valid
                            if (kh_conv >= 0 && kh_conv < kernel_size && kw_conv >= 0 && kw_conv < kernel_size) {
                                // Process input channels in this tile
                                #pragma unroll 4
                                for (int ic_offset = 0; ic_offset < (ic_end - ic_tile); ic_offset++) {
                                    int ic = ic_tile + ic_offset;
                                    
                                    float input_val = input[b * in_channels * in_height * in_width + 
                                                         ic * in_height * in_width + 
                                                         ih * in_width + 
                                                         iw];
                                    
                                    float weight_val = weights_shared[ic_offset * kernel_size * kernel_size + 
                                                                   kh_conv * kernel_size + 
                                                                   kw_conv];
                                    
                                    conv_val += input_val * weight_val;
                                }
                            }
                        }
                    }
                    __syncthreads();
                }
                
                // Apply hardtanh
                float hardtanh_val = conv_val < hardtanh_min ? hardtanh_min : (conv_val > hardtanh_max ? hardtanh_max : conv_val);
                
                // Update max value
                lane_max = max(lane_max, hardtanh_val);
            }
        }
        
        // Warp-level max reduction
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            float other_max = __shfl_down_sync(0xffffffff, lane_max, offset);
            lane_max = max(lane_max, other_max);
        }
        
        // First lane has the final max value
        if (lane_id == 0) {
            thread_sum += lane_max;
        }
    }
    
    // Store warp's sum in shared memory (only for lane 0 of each warp)
    if (lane_id == 0) {
        partial_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (only by the first warp)
    if (warp_id == 0) {
        float warp_sum = lane_id < num_warps ? partial_sums[lane_id] : 0.0f;
        
        // Warp-level sum reduction
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // Final mean and tanh
        if (lane_id == 0) {
            float mean = warp_sum / (pool_out_height * pool_out_width);
            output[b * out_channels * 1 * 1 + oc] = tanh(mean);
        }
    }
}

// C++ interface
torch::Tensor fused_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int maxpool_kernel_size,
    int maxpool_stride,
    float hardtanh_min,
    float hardtanh_max) {
    
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, 1, 1}, 
                              input.options());
    
    // Launch kernel
    const int block_size = 256;
    const int tile_size = 16;
    dim3 grid(batch_size, out_channels);
    dim3 block(block_size);
    
    // Calculate shared memory size
    const int shared_mem_size = (tile_size * kernel_size * kernel_size + block_size) * sizeof(float);
    
    fused_convtranspose_maxpool_hardtanh_mean_tanh_kernel<block_size, tile_size><<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_size, stride, padding,
        maxpool_kernel_size, maxpool_stride,
        hardtanh_min, hardtanh_max
    );
    
    return output;
}

// C++ interface for warp-optimized version
torch::Tensor warp_optimized_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int maxpool_kernel_size,
    int maxpool_stride,
    float hardtanh_min,
    float hardtanh_max) {
    
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, 1, 1}, 
                              input.options());
    
    // Launch kernel
    const int block_size = 256;
    const int tile_size = 16;
    dim3 grid(batch_size, out_channels);
    dim3 block(block_size);
    
    // Calculate shared memory size
    const int shared_mem_size = (tile_size * kernel_size * kernel_size + (block_size / 32)) * sizeof(float);
    
    warp_optimized_kernel<block_size, tile_size><<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        in_height, in_width,
        kernel_size, stride, padding,
        maxpool_kernel_size, maxpool_stride,
        hardtanh_min, hardtanh_max
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_forward, "Fused ConvTranspose2d + MaxPool + Hardtanh + Mean + Tanh forward");
    m.def("warp_optimized_forward", &warp_optimized_forward, "Warp-optimized fused forward");
}
"""

class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, followed by max pooling, hardtanh activation, mean operation, and tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        
        # Create reference PyTorch modules to initialize weights
        ref_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
        # Create our parameters
        self.weight = nn.Parameter(ref_conv.weight.data.clone())
        self.bias = nn.Parameter(ref_conv.bias.data.clone())
        
        # Compile CUDA extension
        try:
            self.cuda_extension = load_inline(
                name="optimized_ops",
                cpp_sources="",
                cuda_sources=cuda_source,
                functions=["fused_forward", "warp_optimized_forward"],
                with_cuda=True,
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False
            )
            self.use_cuda_extension = True
        except Exception as e:
            print(f"Failed to load CUDA extension: {e}")
            self.use_cuda_extension = False

    def forward(self, x):
        # Fallback to PyTorch implementation if CUDA extension failed to load
        if not self.use_cuda_extension:
            # Use PyTorch's implementation
            x = F.conv_transpose2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
            x = F.max_pool2d(x, kernel_size=self.maxpool_kernel_size, stride=self.maxpool_stride)
            x = F.hardtanh(x, min_val=self.hardtanh_min, max_val=self.hardtanh_max)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x
        
        try:
            # Try using our warp-optimized CUDA kernel
            return self.cuda_extension.warp_optimized_forward(
                x, self.weight, self.bias,
                self.stride, self.padding,
                self.maxpool_kernel_size, self.maxpool_stride,
                self.hardtanh_min, self.hardtanh_max
            )
        except Exception as e:
            print(f"Warp-optimized kernel failed: {e}. Trying basic fused kernel.")
            try:
                # Try using our basic fused CUDA kernel
                return self.cuda_extension.fused_forward(
                    x, self.weight, self.bias,
                    self.stride, self.padding,
                    self.maxpool_kernel_size, self.maxpool_stride,
                    self.hardtanh_min, self.hardtanh_max
                )
            except Exception as e:
                print(f"CUDA kernel execution failed: {e}. Falling back to PyTorch implementation.")
                # Fallback to PyTorch implementation
                x = F.conv_transpose2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
                x = F.max_pool2d(x, kernel_size=self.maxpool_kernel_size, stride=self.maxpool_stride)
                x = F.hardtanh(x, min_val=self.hardtanh_min, max_val=self.hardtanh_max)
                x = torch.mean(x, dim=(2, 3), keepdim=True)
                x = torch.tanh(x)
                return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1
hardtanh_max = 1

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]