import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with asymmetric input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # For depthwise convolution, in_channels should equal out_channels
        assert in_channels == out_channels, "For depthwise convolution, in_channels must equal out_channels"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights similar to nn.Conv2d
        self.weight = nn.Parameter(torch.Tensor(out_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights using the same method as nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Compile CUDA kernel if we're on a CUDA device
        self.kernel_module = None
        if torch.cuda.is_available():
            self._compile_kernel()

    def _compile_kernel(self):
        # Define the CUDA kernel
        cuda_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        
        // Optimized kernel for 3x3 depthwise convolution with stride=1, padding=0
        // Specifically optimized for 3 channels
        template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y, int ITEMS_PER_THREAD_X, int ITEMS_PER_THREAD_Y>
        __global__ void depthwise_conv2d_kernel_optimized(
            const float* __restrict__ input,
            const float* __restrict__ weight,
            float* __restrict__ output,
            const int batch_size,
            const int channels,
            const int height_in,
            const int width_in,
            const int height_out,
            const int width_out)
        {
            // Constants for 3x3 kernel
            constexpr int KERNEL_SIZE = 3;
            constexpr int KERNEL_RADIUS = 1;  // (KERNEL_SIZE - 1) / 2
            
            // Calculate base output position for this thread block
            const int tile_start_x = blockIdx.x * (BLOCK_SIZE_X * ITEMS_PER_THREAD_X);
            const int tile_start_y = blockIdx.y * (BLOCK_SIZE_Y * ITEMS_PER_THREAD_Y);
            
            // Calculate which channel and batch this thread block is processing
            const int c = blockIdx.z % channels;
            const int b = blockIdx.z / channels;
            
            // Define tile dimensions including halo region for the kernel
            constexpr int TILE_WIDTH = BLOCK_SIZE_X * ITEMS_PER_THREAD_X + 2*KERNEL_RADIUS;
            constexpr int TILE_HEIGHT = BLOCK_SIZE_Y * ITEMS_PER_THREAD_Y + 2*KERNEL_RADIUS;
            
            // Define shared memory for input tile with padding for halo regions
            // Add padding to avoid bank conflicts
            __shared__ float s_input[TILE_HEIGHT][TILE_WIDTH + 1];
            
            // Load kernel weights into registers for faster access
            float w[KERNEL_SIZE][KERNEL_SIZE];
            #pragma unroll
            for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                    w[ky][kx] = weight[(c * KERNEL_SIZE + ky) * KERNEL_SIZE + kx];
                }
            }
            
            // Calculate input base offset for this batch and channel
            const int input_batch_offset = (b * channels + c) * height_in * width_in;
            
            // Collaborative loading of input data into shared memory
            // Each thread loads multiple elements to maximize memory bandwidth utilization
            const int thread_idx = threadIdx.y * BLOCK_SIZE_X + threadIdx.x;
            const int num_threads = BLOCK_SIZE_X * BLOCK_SIZE_Y;
            const int total_elements = TILE_HEIGHT * TILE_WIDTH;
            
            // Optimize loading pattern for better coalescing
            // Load in rows for better memory coalescing
            #pragma unroll 2
            for (int row = 0; row < TILE_HEIGHT; ++row) {
                const int y_in = tile_start_y + row - KERNEL_RADIUS;
                
                // Each thread in a row loads multiple elements in a strided pattern
                for (int col = thread_idx; col < TILE_WIDTH; col += num_threads) {
                    const int x_in = tile_start_x + col - KERNEL_RADIUS;
                    
                    float value = 0.0f;
                    if (y_in >= 0 && y_in < height_in && x_in >= 0 && x_in < width_in) {
                        value = input[input_batch_offset + y_in * width_in + x_in];
                    }
                    
                    s_input[row][col] = value;
                }
            }
            
            __syncthreads();
            
            // Calculate output base offset for this batch and channel
            const int output_batch_offset = (b * channels + c) * height_out * width_out;
            
            // Each thread computes multiple output pixels
            #pragma unroll
            for (int y_item = 0; y_item < ITEMS_PER_THREAD_Y; ++y_item) {
                const int out_y = tile_start_y + threadIdx.y * ITEMS_PER_THREAD_Y + y_item;
                
                if (out_y < height_out) {
                    #pragma unroll
                    for (int x_item = 0; x_item < ITEMS_PER_THREAD_X; ++x_item) {
                        const int out_x = tile_start_x + threadIdx.x * ITEMS_PER_THREAD_X + x_item;
                        
                        if (out_x < width_out) {
                            // Calculate the position in shared memory
                            const int s_y = threadIdx.y * ITEMS_PER_THREAD_Y + y_item;
                            const int s_x = threadIdx.x * ITEMS_PER_THREAD_X + x_item;
                            
                            // Compute convolution with fully unrolled operations for 3x3 kernel
                            float sum = 0.0f;
                            
                            // Fully unroll the 3x3 convolution for better performance
                            sum += s_input[s_y + 0][s_x + 0] * w[0][0];
                            sum += s_input[s_y + 0][s_x + 1] * w[0][1];
                            sum += s_input[s_y + 0][s_x + 2] * w[0][2];
                            sum += s_input[s_y + 1][s_x + 0] * w[1][0];
                            sum += s_input[s_y + 1][s_x + 1] * w[1][1];
                            sum += s_input[s_y + 1][s_x + 2] * w[1][2];
                            sum += s_input[s_y + 2][s_x + 0] * w[2][0];
                            sum += s_input[s_y + 2][s_x + 1] * w[2][1];
                            sum += s_input[s_y + 2][s_x + 2] * w[2][2];
                            
                            // Write output
                            output[output_batch_offset + out_y * width_out + out_x] = sum;
                        }
                    }
                }
            }
        }
        
        // Generic kernel for depthwise convolution with arbitrary parameters
        template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
        __global__ void depthwise_conv2d_kernel_generic(
            const float* __restrict__ input,
            const float* __restrict__ weight,
            float* __restrict__ output,
            const int batch_size,
            const int channels,
            const int height_in,
            const int width_in,
            const int height_out,
            const int width_out,
            const int kernel_size,
            const int stride,
            const int padding)
        {
            // Calculate output position
            const int out_x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
            const int out_y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
            const int c = blockIdx.z % channels;
            const int b = blockIdx.z / channels;
            
            // Early exit if outside output dimensions
            if (out_x >= width_out || out_y >= height_out)
                return;
                
            // Compute convolution
            float sum = 0.0f;
            
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    const int in_y = out_y * stride + ky - padding;
                    const int in_x = out_x * stride + kx - padding;
                    
                    if (in_y >= 0 && in_y < height_in && in_x >= 0 && in_x < width_in) {
                        const int input_idx = ((b * channels + c) * height_in + in_y) * width_in + in_x;
                        const int weight_idx = (c * kernel_size + ky) * kernel_size + kx;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
            
            const int output_idx = ((b * channels + c) * height_out + out_y) * width_out + out_x;
            output[output_idx] = sum;
        }
        
        torch::Tensor depthwise_conv2d_cuda(
            torch::Tensor input,
            torch::Tensor weight,
            int kernel_size,
            int stride,
            int padding)
        {
            // Get dimensions
            const int batch_size = input.size(0);
            const int channels = input.size(1);
            const int height_in = input.size(2);
            const int width_in = input.size(3);
            
            // Handle padding if needed
            torch::Tensor padded_input = input;
            int padded_height = height_in;
            int padded_width = width_in;
            
            if (padding > 0) {
                // Create padded input
                padded_input = torch::zeros({batch_size, channels, height_in + 2 * padding, width_in + 2 * padding}, 
                                          input.options());
                padded_input.slice(2, padding, padding + height_in)
                          .slice(3, padding, padding + width_in)
                          .copy_(input);
                          
                padded_height = height_in + 2 * padding;
                padded_width = width_in + 2 * padding;
            }
            
            // Calculate output dimensions
            const int height_out = (padded_height - kernel_size) / stride + 1;
            const int width_out = (padded_width - kernel_size) / stride + 1;
            
            // Create output tensor
            auto output = torch::zeros({batch_size, channels, height_out, width_out}, 
                                      input.options());
            
            // Get pointers to tensor data
            const float* input_ptr = padded_input.data_ptr<float>();
            const float* weight_ptr = weight.data_ptr<float>();
            float* output_ptr = output.data_ptr<float>();
            
            // Optimize for the specific case of 3x3 kernel, stride=1
            if (kernel_size == 3 && stride == 1) {
                // Thread block and grid configuration for optimized kernel
                constexpr int BLOCK_SIZE_X = 32;
                constexpr int BLOCK_SIZE_Y = 8;
                constexpr int ITEMS_PER_THREAD_X = 2;
                constexpr int ITEMS_PER_THREAD_Y = 2;
                
                dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
                dim3 grid(
                    (width_out + BLOCK_SIZE_X * ITEMS_PER_THREAD_X - 1) / (BLOCK_SIZE_X * ITEMS_PER_THREAD_X),
                    (height_out + BLOCK_SIZE_Y * ITEMS_PER_THREAD_Y - 1) / (BLOCK_SIZE_Y * ITEMS_PER_THREAD_Y),
                    batch_size * channels
                );
                
                depthwise_conv2d_kernel_optimized<BLOCK_SIZE_X, BLOCK_SIZE_Y, ITEMS_PER_THREAD_X, ITEMS_PER_THREAD_Y><<<grid, block>>>(
                    input_ptr, weight_ptr, output_ptr,
                    batch_size, channels, padded_height, padded_width, height_out, width_out
                );
            } else {
                // Generic case for other parameters
                constexpr int BLOCK_SIZE_X = 16;
                constexpr int BLOCK_SIZE_Y = 16;
                
                dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
                dim3 grid(
                    (width_out + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                    (height_out + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,
                    batch_size * channels
                );
                
                depthwise_conv2d_kernel_generic<BLOCK_SIZE_X, BLOCK_SIZE_Y><<<grid, block>>>(
                    input_ptr, weight_ptr, output_ptr,
                    batch_size, channels, padded_height, padded_width, height_out, width_out,
                    kernel_size, stride, padding
                );
            }
            
            return output;
        }
        """

        cpp_source = """
        #include <torch/extension.h>
        
        // Forward declaration of CUDA functions
        torch::Tensor depthwise_conv2d_cuda(
            torch::Tensor input,
            torch::Tensor weight,
            int kernel_size,
            int stride,
            int padding);
        
        // Python bindings
        torch::Tensor depthwise_conv2d(
            torch::Tensor input,
            torch::Tensor weight,
            int kernel_size,
            int stride,
            int padding) {
            
            return depthwise_conv2d_cuda(input, weight, kernel_size, stride, padding);
        }
        
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("depthwise_conv2d", &depthwise_conv2d, "Depthwise Convolution 2D");
        }
        """

        try:
            from torch.utils.cpp_extension import load_inline
            self.kernel_module = load_inline(
                name='depthwise_conv2d_opt',
                cpp_sources=cpp_source,
                cuda_sources=cuda_source,
                functions=['depthwise_conv2d'],
                verbose=False
            )
        except Exception as e:
            print(f"Warning: Failed to compile CUDA kernel: {e}")
            self.kernel_module = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # If we have a working CUDA kernel and the input is on CUDA
        if self.kernel_module is not None and x.is_cuda:
            try:
                # Ensure input is contiguous
                x = x.contiguous()
                
                # Reshape weight for depthwise convolution
                weight = self.weight.view(self.out_channels, self.kernel_size, self.kernel_size).contiguous()
                
                # Call our optimized CUDA kernel
                output = self.kernel_module.depthwise_conv2d(
                    x, weight, self.kernel_size, self.stride, self.padding
                )
                
                # Add bias if needed
                if self.bias is not None:
                    output += self.bias.view(1, -1, 1, 1)
                
                return output
            except Exception as e:
                print(f"Warning: CUDA kernel failed: {e}. Falling back to PyTorch implementation.")
        
        # Fallback to PyTorch's implementation
        return nn.functional.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, 1, self.in_channels
        )

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
out_channels = 3
kernel_size = 3
width_in = 256
height_in = 128
stride = 1
padding = 0

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]