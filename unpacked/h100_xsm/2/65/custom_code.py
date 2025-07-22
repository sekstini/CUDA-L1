import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline
import os

class ModelNew(nn.Module):
    """
    Optimized implementation of the model using custom CUDA kernels
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolution kernel
        pool_kernel_size (int): Size of the pooling kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        
        # Initialize weights and bias like PyTorch's Conv2d
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        
        # Define CUDA kernel for fully fused operations
        cuda_source = '''
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <vector>

        template <typename scalar_t>
        __device__ __forceinline__ scalar_t sigmoid_fast(scalar_t x) {
            return scalar_t(1) / (scalar_t(1) + exp(-x));
        }

        // CUDA kernel for fully fused convolution, pooling, sigmoid, and sum
        template <typename scalar_t>
        __global__ void fused_conv_pool_sigmoid_sum_kernel(
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
            const int pool_size) {
            
            // Calculate output dimensions
            const int conv_out_height = in_height - kernel_size + 1;
            const int conv_out_width = in_width - kernel_size + 1;
            const int pool_out_height = conv_out_height / pool_size;
            const int pool_out_width = conv_out_width / pool_size;
            
            // Define shared memory
            extern __shared__ char shared_mem_char[];
            scalar_t* shared_mem = reinterpret_cast<scalar_t*>(shared_mem_char);
            
            // Each thread block handles one batch element
            const int batch_idx = blockIdx.x;
            
            // Thread indices
            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            const int tid = ty * blockDim.x + tx;
            const int block_size = blockDim.x * blockDim.y;
            
            // Calculate tile dimensions for input data caching
            const int tile_width = 16;  // Match blockDim.x
            const int tile_height = 16; // Match blockDim.y
            
            // Partition shared memory
            scalar_t* s_weight = &shared_mem[0];
            scalar_t* s_bias = &s_weight[out_channels * in_channels * kernel_size * kernel_size];
            scalar_t* s_input_tile = &s_bias[out_channels];
            scalar_t* s_partial_sums = &s_input_tile[in_channels * (tile_height + kernel_size - 1) * (tile_width + kernel_size - 1)];
            
            // Load weights into shared memory - collaborative loading
            const int weights_per_thread = (out_channels * in_channels * kernel_size * kernel_size + block_size - 1) / block_size;
            
            for (int i = 0; i < weights_per_thread; i++) {
                const int weight_idx = tid + i * block_size;
                if (weight_idx < out_channels * in_channels * kernel_size * kernel_size) {
                    s_weight[weight_idx] = weight[weight_idx];
                }
            }
            
            // Load bias into shared memory
            if (tid < out_channels) {
                s_bias[tid] = bias[tid];
            }
            
            __syncthreads();
            
            // Initialize thread's partial sum
            scalar_t thread_sum = 0.0f;
            
            // Process output in tiles
            for (int tile_y = 0; tile_y < pool_out_height; tile_y += tile_height / pool_size) {
                for (int tile_x = 0; tile_x < pool_out_width; tile_x += tile_width / pool_size) {
                    
                    // Calculate input tile coordinates
                    const int in_tile_y = tile_y * pool_size;
                    const int in_tile_x = tile_x * pool_size;
                    const int in_tile_height = min(tile_height, (pool_out_height - tile_y) * pool_size) + kernel_size - 1;
                    const int in_tile_width = min(tile_width, (pool_out_width - tile_x) * pool_size) + kernel_size - 1;
                    
                    // Collaborative loading of input tile into shared memory
                    const int input_elements_per_thread = (in_channels * in_tile_height * in_tile_width + block_size - 1) / block_size;
                    
                    for (int i = 0; i < input_elements_per_thread; i++) {
                        const int input_idx = tid + i * block_size;
                        if (input_idx < in_channels * in_tile_height * in_tile_width) {
                            const int ic = input_idx / (in_tile_height * in_tile_width);
                            const int remainder = input_idx % (in_tile_height * in_tile_width);
                            const int iy = remainder / in_tile_width;
                            const int ix = remainder % in_tile_width;
                            
                            const int global_y = in_tile_y + iy;
                            const int global_x = in_tile_x + ix;
                            
                            if (global_y < in_height && global_x < in_width) {
                                const int in_idx = ((batch_idx * in_channels + ic) * in_height + global_y) * in_width + global_x;
                                s_input_tile[input_idx] = input[in_idx];
                            } else {
                                s_input_tile[input_idx] = 0.0f;
                            }
                        }
                    }
                    
                    __syncthreads();
                    
                    // Process output elements in this tile
                    const int out_tile_height = min(tile_height / pool_size, pool_out_height - tile_y);
                    const int out_tile_width = min(tile_width / pool_size, pool_out_width - tile_x);
                    
                    // Each thread processes multiple output elements
                    const int elements_per_thread = (out_tile_height * out_tile_width * out_channels + block_size - 1) / block_size;
                    
                    for (int i = 0; i < elements_per_thread; i++) {
                        const int elem_idx = tid + i * block_size;
                        if (elem_idx < out_tile_height * out_tile_width * out_channels) {
                            const int oc = elem_idx / (out_tile_height * out_tile_width);
                            const int remainder = elem_idx % (out_tile_height * out_tile_width);
                            const int local_oh = remainder / out_tile_width;
                            const int local_ow = remainder % out_tile_width;
                            
                            const int oh = tile_y + local_oh;
                            const int ow = tile_x + local_ow;
                            
                            if (oh < pool_out_height && ow < pool_out_width && oc < out_channels) {
                                // Calculate the convolution output region for this pooling cell
                                const int conv_y_start = oh * pool_size;
                                const int conv_x_start = ow * pool_size;
                                
                                // Initialize accumulator for pooling
                                scalar_t pool_sum = 0.0f;
                                int valid_pool_elements = 0;
                                
                                // Perform convolution and pooling
                                for (int py = 0; py < pool_size; ++py) {
                                    for (int px = 0; px < pool_size; ++px) {
                                        const int conv_y = conv_y_start + py;
                                        const int conv_x = conv_x_start + px;
                                        
                                        // Skip if outside valid convolution output area
                                        if (conv_y >= conv_out_height || conv_x >= conv_out_width)
                                            continue;
                                        
                                        valid_pool_elements++;
                                        
                                        // Initialize convolution result with bias
                                        scalar_t conv_result = s_bias[oc];
                                        
                                        // Perform convolution for this position - specialized for 3x3 kernel
                                        if (kernel_size == 3) {
                                            for (int ic = 0; ic < in_channels; ++ic) {
                                                // Calculate offsets in the input tile
                                                const int local_y = conv_y - in_tile_y;
                                                const int local_x = conv_x - in_tile_x;
                                                const int in_tile_offset = (ic * in_tile_height + local_y) * in_tile_width + local_x;
                                                const int w_offset = ((oc * in_channels + ic) * kernel_size) * kernel_size;
                                                
                                                // Fully unrolled 3x3 convolution using cached input data
                                                conv_result += s_input_tile[in_tile_offset] * s_weight[w_offset];
                                                conv_result += s_input_tile[in_tile_offset + 1] * s_weight[w_offset + 1];
                                                conv_result += s_input_tile[in_tile_offset + 2] * s_weight[w_offset + 2];
                                                
                                                conv_result += s_input_tile[in_tile_offset + in_tile_width] * s_weight[w_offset + 3];
                                                conv_result += s_input_tile[in_tile_offset + in_tile_width + 1] * s_weight[w_offset + 4];
                                                conv_result += s_input_tile[in_tile_offset + in_tile_width + 2] * s_weight[w_offset + 5];
                                                
                                                conv_result += s_input_tile[in_tile_offset + 2*in_tile_width] * s_weight[w_offset + 6];
                                                conv_result += s_input_tile[in_tile_offset + 2*in_tile_width + 1] * s_weight[w_offset + 7];
                                                conv_result += s_input_tile[in_tile_offset + 2*in_tile_width + 2] * s_weight[w_offset + 8];
                                            }
                                        } else {
                                            // Generic case for any kernel size
                                            for (int ic = 0; ic < in_channels; ++ic) {
                                                for (int ky = 0; ky < kernel_size; ++ky) {
                                                    for (int kx = 0; kx < kernel_size; ++kx) {
                                                        // Calculate offsets in the input tile
                                                        const int local_y = conv_y - in_tile_y + ky;
                                                        const int local_x = conv_x - in_tile_x + kx;
                                                        const int in_tile_idx = (ic * in_tile_height + local_y) * in_tile_width + local_x;
                                                        const int w_idx = ((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
                                                        
                                                        conv_result += s_input_tile[in_tile_idx] * s_weight[w_idx];
                                                    }
                                                }
                                            }
                                        }
                                        
                                        // Add to pooling sum
                                        pool_sum += conv_result;
                                    }
                                }
                                
                                // Average pooling
                                if (valid_pool_elements > 0) {
                                    pool_sum /= valid_pool_elements;
                                }
                                
                                // Apply sigmoid
                                pool_sum = sigmoid_fast(pool_sum);
                                
                                // Add to thread's partial sum
                                thread_sum += pool_sum;
                            }
                        }
                    }
                    
                    __syncthreads();
                }
            }
            
            // Store partial sum in shared memory
            s_partial_sums[tid] = thread_sum;
            __syncthreads();
            
            // Parallel reduction in shared memory
            for (int stride = block_size / 2; stride > 32; stride >>= 1) {
                if (tid < stride) {
                    s_partial_sums[tid] += s_partial_sums[tid + stride];
                }
                __syncthreads();
            }
            
            // Warp-level reduction (more efficient for the last steps)
            if (tid < 32) {
                // Volatile operations for warp synchronization
                volatile scalar_t* smem = s_partial_sums;
                if (block_size >= 64) smem[tid] += smem[tid + 32];
                if (block_size >= 32) smem[tid] += smem[tid + 16];
                if (block_size >= 16) smem[tid] += smem[tid + 8];
                if (block_size >= 8) smem[tid] += smem[tid + 4];
                if (block_size >= 4) smem[tid] += smem[tid + 2];
                if (block_size >= 2) smem[tid] += smem[tid + 1];
            }
            
            // First thread writes the final result for this batch element
            if (tid == 0) {
                output[batch_idx] = s_partial_sums[0];
            }
        }

        std::vector<torch::Tensor> fused_conv_pool_sigmoid_sum_cuda_forward(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
            int pool_size) {
            
            const auto batch_size = input.size(0);
            const auto in_channels = input.size(1);
            const auto in_height = input.size(2);
            const auto in_width = input.size(3);
            const auto out_channels = weight.size(0);
            const auto kernel_size = weight.size(2);
            
            auto output = torch::zeros({batch_size}, input.options());
            
            // Configure kernel launch parameters
            const int block_dim_x = 16;
            const int block_dim_y = 16;
            const dim3 block_dim(block_dim_x, block_dim_y);
            
            // Grid dimensions - one block per batch element
            const int grid_dim_x = batch_size;
            const dim3 grid_dim(grid_dim_x);
            
            // Calculate shared memory size
            const int tile_width = block_dim_x;
            const int tile_height = block_dim_y;
            const int weight_size = out_channels * in_channels * kernel_size * kernel_size;
            const int bias_size = out_channels;
            const int input_tile_size = in_channels * (tile_height + kernel_size - 1) * (tile_width + kernel_size - 1);
            const int partial_sums_size = block_dim_x * block_dim_y;
            
            const int shared_mem_size = (weight_size + bias_size + input_tile_size + partial_sums_size) * sizeof(float);
            
            AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_conv_pool_sigmoid_sum_cuda_forward", ([&] {
                fused_conv_pool_sigmoid_sum_kernel<scalar_t><<<grid_dim, block_dim, shared_mem_size>>>(
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    in_channels,
                    out_channels,
                    in_height,
                    in_width,
                    kernel_size,
                    pool_size
                );
            }));
            
            return {output};
        }
        '''

        cpp_source = '''
        #include <torch/extension.h>
        #include <vector>

        // CUDA forward declarations
        std::vector<torch::Tensor> fused_conv_pool_sigmoid_sum_cuda_forward(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
            int pool_size);

        // C++ interface
        std::vector<torch::Tensor> fused_conv_pool_sigmoid_sum_forward(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
            int pool_size) {
            return fused_conv_pool_sigmoid_sum_cuda_forward(input, weight, bias, pool_size);
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("forward", &fused_conv_pool_sigmoid_sum_forward, "Fused Conv2d+AvgPool+Sigmoid+Sum forward (CUDA)");
        }
        '''
        
        # Compile CUDA kernel
        self.cuda_module = None
        try:
            # Only compile if not already compiled
            if not os.path.exists('optimized_conv_cuda.so'):
                self.cuda_module = load_inline(
                    name='optimized_conv_cuda',
                    cpp_sources=cpp_source,
                    cuda_sources=cuda_source,
                    functions=['forward'],
                    verbose=True,
                    with_cuda=True
                )
            else:
                import importlib.util
                spec = importlib.util.spec_from_file_location("optimized_conv_cuda", "./optimized_conv_cuda.so")
                if spec is not None:
                    self.cuda_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(self.cuda_module)
                else:
                    raise ImportError("Failed to import optimized_conv_cuda.so")
        except Exception as e:
            print(f"CUDA kernel compilation failed: {e}")
            self.cuda_module = None
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        if self.cuda_module is not None and x.is_cuda:
            try:
                return self.cuda_module.forward(x, self.weight, self.bias, self.pool_kernel_size)[0]
            except Exception as e:
                print(f"CUDA kernel execution failed: {e}, falling back to PyTorch implementation")
        
        # Fallback to PyTorch implementation
        x = F.conv2d(x, self.weight, self.bias)
        x = F.avg_pool2d(x, self.pool_kernel_size)
        x = torch.sigmoid(x)
        x = torch.sum(x, dim=[1,2,3])
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, pool_kernel_size]