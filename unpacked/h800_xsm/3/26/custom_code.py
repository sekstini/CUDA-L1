import torch
import torch.nn as nn
import torch.nn.functional as F

# CUDA extension for optimized channel shuffle
channel_shuffle_cuda_code = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Helper function to check CUDA errors
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s at %s:%d\\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        throw std::runtime_error("CUDA error"); \
    } \
}

// Optimized kernel for small feature maps
template <typename scalar_t>
__global__ void channel_shuffle_small_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int groups,
    const int channels_per_group) {
    
    // Each thread handles one spatial position across all channels
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_spatial = batch_size * height * width;
    
    if (idx < total_spatial) {
        const int b = idx / (height * width);
        const int h = (idx % (height * width)) / width;
        const int w = idx % width;
        
        // Process all channels for this spatial location
        #pragma unroll 4
        for (int c = 0; c < channels; c++) {
            const int group_idx = c / channels_per_group;
            const int channel_in_group = c % channels_per_group;
            const int shuffled_c = channel_in_group * groups + group_idx;
            
            const int input_idx = ((b * channels + c) * height + h) * width + w;
            const int output_idx = ((b * channels + shuffled_c) * height + h) * width + w;
            
            output[output_idx] = input[input_idx];
        }
    }
}

// Ultra-specialized kernel for ShuffleNet with groups=3 and common dimensions
template <typename scalar_t>
__global__ void channel_shuffle_shufflenet_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int height,
    const int width) {
    
    // Calculate spatial position - each thread handles multiple pixels
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = gridDim.x * blockDim.x;
    const int total_pixels = batch_size * height * width;
    
    // Each thread processes multiple pixels
    for (int pixel_idx = thread_id; pixel_idx < total_pixels; pixel_idx += num_threads) {
        const int b = pixel_idx / (height * width);
        const int h = (pixel_idx % (height * width)) / width;
        const int w = pixel_idx % width;
        
        const int channels_per_group = channels / 3;
        const int spatial_idx = h * width + w;
        const int batch_offset = b * channels * height * width;
        
        // Process all channels for this pixel
        for (int g = 0; g < 3; g++) {
            for (int c = 0; c < channels_per_group; c++) {
                const int input_c = g * channels_per_group + c;
                const int output_c = c * 3 + g;
                
                const int input_idx = batch_offset + (input_c * height * width) + spatial_idx;
                const int output_idx = batch_offset + (output_c * height * width) + spatial_idx;
                
                output[output_idx] = input[input_idx];
            }
        }
    }
}

// Specialized kernel for groups=3 (common case in ShuffleNet)
template <typename scalar_t>
__global__ void channel_shuffle_g3_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int height,
    const int width) {
    
    // Calculate spatial position
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    
    if (x < width && y < height) {
        const int channels_per_group = channels / 3;
        
        // Pre-compute base index for better memory access
        const int spatial_idx = y * width + x;
        const int batch_offset = b * channels * height * width;
        
        // Process channels with hardcoded groups=3
        #pragma unroll 4
        for (int c = 0; c < channels; c++) {
            // Calculate group and channel within group
            const int group_idx = c / channels_per_group;
            const int channel_in_group = c % channels_per_group;
            
            // Calculate shuffled channel index for groups=3
            const int shuffled_c = channel_in_group * 3 + group_idx;
            
            // Calculate input and output indices
            const int input_idx = batch_offset + (c * height * width) + spatial_idx;
            const int output_idx = batch_offset + (shuffled_c * height * width) + spatial_idx;
            
            // Copy with shuffled channel index
            output[output_idx] = input[input_idx];
        }
    }
}

// Vector-based kernel for medium-sized feature maps
template <typename scalar_t>
__global__ void channel_shuffle_medium_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int groups,
    const int channels_per_group) {
    
    // Process multiple elements per thread for better throughput
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    const int b = blockIdx.z;
    
    if (x < width && y < height) {
        // Pre-compute base indices
        const int spatial_idx = y * width + x;
        const int batch_offset = b * channels * height * width;
        
        // Process 4 channels at a time where possible
        for (int c = 0; c < channels; c += 4) {
            if (c + 4 <= channels) {
                // Load 4 channels at once
                float4 input_data;
                float* input_ptr = (float*)&input_data;
                
                for (int i = 0; i < 4; i++) {
                    const int curr_c = c + i;
                    const int input_idx = batch_offset + (curr_c * height * width) + spatial_idx;
                    input_ptr[i] = static_cast<float>(input[input_idx]);
                }
                
                // Compute shuffled indices and store
                for (int i = 0; i < 4; i++) {
                    const int curr_c = c + i;
                    const int group_idx = curr_c / channels_per_group;
                    const int channel_in_group = curr_c % channels_per_group;
                    const int shuffled_c = channel_in_group * groups + group_idx;
                    
                    const int output_idx = batch_offset + (shuffled_c * height * width) + spatial_idx;
                    output[output_idx] = static_cast<scalar_t>(input_ptr[i]);
                }
            } else {
                // Handle remaining channels individually
                for (int i = 0; i < min(4, channels - c); i++) {
                    const int curr_c = c + i;
                    const int group_idx = curr_c / channels_per_group;
                    const int channel_in_group = curr_c % channels_per_group;
                    const int shuffled_c = channel_in_group * groups + group_idx;
                    
                    const int input_idx = batch_offset + (curr_c * height * width) + spatial_idx;
                    const int output_idx = batch_offset + (shuffled_c * height * width) + spatial_idx;
                    
                    output[output_idx] = input[input_idx];
                }
            }
        }
    }
}

// Shared memory optimized kernel for larger feature maps
template <typename scalar_t>
__global__ void channel_shuffle_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int groups,
    const int channels_per_group) {
    
    extern __shared__ unsigned char shared_mem[];
    scalar_t* tile = reinterpret_cast<scalar_t*>(shared_mem);
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    
    if (x < width && y < height) {
        // Process channels in tiles to utilize shared memory
        const int tile_size = 32; // Process 32 channels at a time
        const int padded_tile_width = blockDim.x + (blockDim.x % 32 == 0 ? 0 : 32 - (blockDim.x % 32)); // Pad to avoid bank conflicts
        
        for (int c_start = 0; c_start < channels; c_start += tile_size) {
            const int c_end = min(c_start + tile_size, channels);
            
            // Load channel tile into shared memory
            for (int c = c_start + threadIdx.y; c < c_end; c += blockDim.y) {
                if (c < channels && x < width && y < height) {
                    const int input_idx = ((b * channels + c) * height + y) * width + x;
                    // Use padded index to avoid bank conflicts
                    const int smem_idx = (c - c_start) * padded_tile_width + threadIdx.x;
                    tile[smem_idx] = input[input_idx];
                }
            }
            
            __syncthreads();
            
            // Process and write output with shuffled indices
            for (int c = c_start + threadIdx.y; c < c_end; c += blockDim.y) {
                if (c < channels && x < width && y < height) {
                    const int group_idx = c / channels_per_group;
                    const int channel_in_group = c % channels_per_group;
                    const int shuffled_c = channel_in_group * groups + group_idx;
                    
                    const int output_idx = ((b * channels + shuffled_c) * height + y) * width + x;
                    // Use padded index to avoid bank conflicts
                    const int smem_idx = (c - c_start) * padded_tile_width + threadIdx.x;
                    output[output_idx] = tile[smem_idx];
                }
            }
            
            __syncthreads();
        }
    }
}

torch::Tensor channel_shuffle_cuda_forward(
    torch::Tensor input,
    int groups) {
    
    const auto batch_size = input.size(0);
    const auto channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    const int channels_per_group = channels / groups;
    
    auto output = torch::empty_like(input);
    
    // Choose kernel based on tensor dimensions and groups
    const int total_spatial = batch_size * height * width;
    
    // For very small tensors
    if (total_spatial < 1024) {
        const int threads = 256;
        const int blocks = (total_spatial + threads - 1) / threads;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "channel_shuffle_small_kernel", ([&] {
            channel_shuffle_small_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                channels,
                height,
                width,
                groups,
                channels_per_group
            );
        }));
    }
    // For ShuffleNet with groups=3 and large feature maps (ultra-optimized kernel)
    else if (groups == 3 && height >= 56 && width >= 56) {
        const int threads = 256;
        const int blocks = min(65535, (total_spatial + threads - 1) / threads);
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "channel_shuffle_shufflenet_kernel", ([&] {
            channel_shuffle_shufflenet_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                channels,
                height,
                width
            );
        }));
    }
    // For groups=3 (common case in ShuffleNet)
    else if (groups == 3) {
        const dim3 threads(16, 16);
        const dim3 blocks(
            (width + threads.x - 1) / threads.x,
            (height + threads.y - 1) / threads.y,
            batch_size
        );
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "channel_shuffle_g3_kernel", ([&] {
            channel_shuffle_g3_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                channels,
                height,
                width
            );
        }));
    }
    // For medium-sized tensors, use vectorized kernel
    else if (height * width < 4096) {
        const int threads = 32;
        const dim3 blocks(
            (width + threads - 1) / threads,
            height,
            batch_size
        );
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "channel_shuffle_medium_kernel", ([&] {
            channel_shuffle_medium_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                channels,
                height,
                width,
                groups,
                channels_per_group
            );
        }));
    }
    // For larger tensors, use shared memory kernel
    else {
        const dim3 threads(32, 8);
        const dim3 blocks(
            (width + threads.x - 1) / threads.x,
            (height + threads.y - 1) / threads.y,
            batch_size
        );
        
        // Calculate shared memory size with padding to avoid bank conflicts
        const int tile_size = 32;
        const int padded_tile_width = threads.x + (threads.x % 32 == 0 ? 0 : 32 - (threads.x % 32));
        const int smem_size = tile_size * padded_tile_width * sizeof(float);
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "channel_shuffle_shared_kernel", ([&] {
            channel_shuffle_shared_kernel<scalar_t><<<blocks, threads, smem_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                channels,
                height,
                width,
                groups,
                channels_per_group
            );
        }));
    }
    
    // Check for CUDA errors
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &channel_shuffle_cuda_forward, "Channel Shuffle forward (CUDA)");
}
'''

class OptimizedChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(OptimizedChannelShuffle, self).__init__()
        self.groups = groups
        self.cuda_kernel_loaded = False
        self.indices_cache = {}
        
        # Try to load the CUDA extension
        if torch.cuda.is_available():
            try:
                from torch.utils.cpp_extension import load_inline
                self.channel_shuffle_cuda = load_inline(
                    name="channel_shuffle_cuda",
                    cpp_sources="",
                    cuda_sources=channel_shuffle_cuda_code,
                    functions=["forward"],
                    verbose=False
                )
                self.cuda_kernel_loaded = True
            except Exception as e:
                print(f"Failed to load CUDA extension: {e}")
                self.cuda_kernel_loaded = False
    
    def _get_indices(self, channels, device):
        # Cache indices for reuse
        key = (channels, self.groups, str(device))
        if key in self.indices_cache:
            return self.indices_cache[key]
        
        # Compute the shuffled indices
        channels_per_group = channels // self.groups
        indices = torch.arange(channels, device=device)
        indices = indices.view(self.groups, channels_per_group).t().contiguous().view(-1)
        
        # Cache for future use
        self.indices_cache[key] = indices
        return indices
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        
        # Use CUDA kernel if available and tensor is on CUDA
        if self.cuda_kernel_loaded and x.is_cuda:
            try:
                return self.channel_shuffle_cuda.forward(x, self.groups)
            except Exception as e:
                # Fall back to optimized PyTorch implementation
                pass
        
        # Optimized PyTorch implementation using index_select
        indices = self._get_indices(channels, x.device)
        
        # Use index_select for the channel shuffle
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = x.reshape(batch_size * height * width, channels)
        x = torch.index_select(x, 1, indices)
        x = x.view(batch_size, height, width, channels)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        return x

class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        """
        ShuffleNet unit implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param groups: Number of groups for group convolution.
        """
        super(ShuffleNetUnit, self).__init__()
        
        # Ensure the output channels are divisible by groups
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        # First 1x1 group convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Depthwise 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Second 1x1 group convolution
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shuffle operation
        self.shuffle = OptimizedChannelShuffle(groups)
        
        # Shortcut connection if input and output channels are the same
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass for ShuffleNet unit.

        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = F.relu(self.bn3(self.conv3(out)))
        
        out += self.shortcut(x)
        return out

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        """
        ShuffleNet architecture.

        :param num_classes: Number of output classes.
        :param groups: Number of groups for group convolution.
        :param stages_repeats: List of ints specifying the number of repeats for each stage.
        :param stages_out_channels: List of ints specifying the output channels for each stage.
        """
        super(ModelNew, self).__init__()
        
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        
        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.fc = nn.Linear(1024, num_classes)
    
    def _make_stage(self, in_channels, out_channels, repeats, groups):
        """
        Helper function to create a stage of ShuffleNet units.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param repeats: Number of ShuffleNet units in the stage.
        :param groups: Number of groups for group convolution.
        :return: nn.Sequential containing the stage.
        """
        layers = []
        layers.append(ShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for ShuffleNet.

        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes]