import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolution kernel
        subtract_value_1 (float): First value to subtract
        subtract_value_2 (float): Second value to subtract
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        
        # Create weight parameter
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        
        # Create bias parameter
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # Initialize parameters using the same approach as nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Pre-subtract the combined subtraction values from the bias
        self.bias.data.sub_(subtract_value_1 + subtract_value_2)
        
        # Initialize CUDA kernel if available
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            try:
                import cupy as cp
                self.has_cupy = True
                
                # Define CUDA kernel for fused convolution + mish with optimized shared memory
                self.kernel_code = '''
                extern "C" __global__ void fused_conv2d_mish_kernel(
                    const float* __restrict__ input,
                    const float* __restrict__ weight,
                    const float* __restrict__ bias,
                    float* __restrict__ output,
                    const int batch_size,
                    const int in_channels,
                    const int out_channels,
                    const int in_height,
                    const int in_width,
                    const int kernel_size,
                    const int out_height,
                    const int out_width)
                {
                    // Block and thread indices
                    const int tx = threadIdx.x;
                    const int ty = threadIdx.y;
                    const int bx = blockIdx.x;
                    const int by = blockIdx.y;
                    const int bz = blockIdx.z;
                    
                    // Block dimensions
                    const int BLOCK_SIZE_X = 8;
                    const int BLOCK_SIZE_Y = 8;
                    
                    // Each thread computes a 4x4 output tile
                    const int REG_BLOCK_X = 4;
                    const int REG_BLOCK_Y = 4;
                    
                    // Output position (top-left corner of the 4x4 tile)
                    const int x_out_base = (bx * BLOCK_SIZE_X + tx) * REG_BLOCK_X;
                    const int y_out_base = (by * BLOCK_SIZE_Y + ty) * REG_BLOCK_Y;
                    
                    // Batch and channel indices
                    const int c_out = bz % out_channels;
                    const int b = bz / out_channels;
                    
                    // Early exit if completely out of bounds
                    if (x_out_base >= out_width || y_out_base >= out_height || b >= batch_size)
                        return;
                    
                    // Define shared memory for input tile with padding to avoid bank conflicts
                    // For 8x8 block with 4x4 register blocking and 3x3 kernel, we need (8*4+3-1)x(8*4+3-1) = 34x34
                    extern __shared__ float s_input[];
                    
                    // Calculate padded shared memory dimensions
                    const int s_width = BLOCK_SIZE_X * REG_BLOCK_X + kernel_size - 1;
                    
                    // Define registers for output values (4x4 tile)
                    float values[REG_BLOCK_Y][REG_BLOCK_X];
                    
                    // Load bias into registers and initialize values
                    const float bias_val = bias[c_out];
                    
                    #pragma unroll
                    for (int ry = 0; ry < REG_BLOCK_Y; ++ry) {
                        #pragma unroll
                        for (int rx = 0; rx < REG_BLOCK_X; ++rx) {
                            values[ry][rx] = bias_val;
                        }
                    }
                    
                    // Calculate input tile dimensions
                    const int in_tile_start_y = by * BLOCK_SIZE_Y * REG_BLOCK_Y;
                    const int in_tile_start_x = bx * BLOCK_SIZE_X * REG_BLOCK_X;
                    const int in_tile_height = BLOCK_SIZE_Y * REG_BLOCK_Y + kernel_size - 1;
                    const int in_tile_width = BLOCK_SIZE_X * REG_BLOCK_X + kernel_size - 1;
                    
                    // Perform convolution with shared memory
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        // Collaborative loading of input tile into shared memory
                        // Each thread loads multiple elements to maximize utilization
                        for (int i = 0; i < in_tile_height; i += BLOCK_SIZE_Y) {
                            const int load_y = ty + i;
                            if (load_y < in_tile_height) {
                                for (int j = 0; j < in_tile_width; j += BLOCK_SIZE_X) {
                                    const int load_x = tx + j;
                                    if (load_x < in_tile_width) {
                                        const int y_in = in_tile_start_y + load_y;
                                        const int x_in = in_tile_start_x + load_x;
                                        
                                        float val = 0.0f;
                                        if (y_in < in_height && x_in < in_width && y_in >= 0 && x_in >= 0) {
                                            val = input[((b * in_channels + c_in) * in_height + y_in) * in_width + x_in];
                                        }
                                        
                                        // Store in shared memory using a linear layout to avoid bank conflicts
                                        s_input[load_y * s_width + load_x] = val;
                                    }
                                }
                            }
                        }
                        
                        // Synchronize to make sure all threads have loaded their part of the input
                        __syncthreads();
                        
                        // Perform convolution for this input channel with register blocking
                        #pragma unroll
                        for (int ry = 0; ry < REG_BLOCK_Y; ++ry) {
                            const int y_out = y_out_base + ry;
                            if (y_out < out_height) {
                                #pragma unroll
                                for (int rx = 0; rx < REG_BLOCK_X; ++rx) {
                                    const int x_out = x_out_base + rx;
                                    if (x_out < out_width) {
                                        // Calculate the position in the input tile
                                        const int y_in_local = ty * REG_BLOCK_Y + ry;
                                        const int x_in_local = tx * REG_BLOCK_X + rx;
                                        
                                        // Compute convolution for this output element
                                        #pragma unroll
                                        for (int kh = 0; kh < kernel_size; ++kh) {
                                            #pragma unroll
                                            for (int kw = 0; kw < kernel_size; ++kw) {
                                                const int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                                                const int s_idx = (y_in_local + kh) * s_width + (x_in_local + kw);
                                                values[ry][rx] += s_input[s_idx] * weight[weight_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Synchronize before loading next channel
                        __syncthreads();
                    }
                    
                    // Apply Mish activation and write output for each element in the register block
                    #pragma unroll
                    for (int ry = 0; ry < REG_BLOCK_Y; ++ry) {
                        const int y_out = y_out_base + ry;
                        if (y_out < out_height) {
                            #pragma unroll
                            for (int rx = 0; rx < REG_BLOCK_X; ++rx) {
                                const int x_out = x_out_base + rx;
                                if (x_out < out_width) {
                                    // Optimized Mish activation: x * tanh(softplus(x))
                                    float x = values[ry][rx];
                                    float softplus_val;
                                    
                                    // Optimized softplus calculation
                                    if (x > 20.0f) {
                                        // For large values, softplus(x) ≈ x to avoid overflow
                                        softplus_val = x;
                                    } else if (x < -20.0f) {
                                        // For very negative values, softplus(x) ≈ exp(x)
                                        softplus_val = expf(x);
                                    } else {
                                        softplus_val = logf(1.0f + expf(x));
                                    }
                                    
                                    // Optimized tanh calculation
                                    float tanh_val;
                                    if (softplus_val > 10.0f) {
                                        tanh_val = 1.0f;
                                    } else if (softplus_val < -10.0f) {
                                        tanh_val = -1.0f;
                                    } else {
                                        float exp2x = expf(2.0f * softplus_val);
                                        tanh_val = (exp2x - 1.0f) / (exp2x + 1.0f);
                                    }
                                    
                                    float mish_val = x * tanh_val;
                                    
                                    // Write output
                                    const int output_idx = ((b * out_channels + c_out) * out_height + y_out) * out_width + x_out;
                                    output[output_idx] = mish_val;
                                }
                            }
                        }
                    }
                }
                '''
                
                # Compile the kernel
                self.cuda_module = cp.RawModule(code=self.kernel_code)
                self.fused_kernel = self.cuda_module.get_function("fused_conv2d_mish_kernel")
                
            except ImportError:
                self.has_cupy = False
        else:
            self.has_cupy = False

    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor after convolution, subtraction, and Mish activation
        """
        # Use our custom CUDA kernel if available and input is on CUDA
        if self.use_cuda and self.has_cupy and x.is_cuda:
            try:
                import cupy as cp
                
                # Ensure input is contiguous for better memory access
                if not x.is_contiguous():
                    x = x.contiguous()
                
                batch_size, in_channels, in_height, in_width = x.shape
                out_height = in_height - self.kernel_size + 1
                out_width = in_width - self.kernel_size + 1
                
                # Create output tensor
                output = torch.empty(batch_size, self.out_channels, out_height, out_width, 
                                    device=x.device, dtype=x.dtype)
                
                # Calculate grid and block dimensions
                # Using 8x8 thread blocks with 4x4 register blocking
                threads_per_block_x = 8
                threads_per_block_y = 8
                reg_block_x = 4
                reg_block_y = 4
                
                blocks_x = (out_width + threads_per_block_x * reg_block_x - 1) // (threads_per_block_x * reg_block_x)
                blocks_y = (out_height + threads_per_block_y * reg_block_y - 1) // (threads_per_block_y * reg_block_y)
                blocks_z = batch_size * self.out_channels
                
                # Calculate shared memory size
                s_width = threads_per_block_x * reg_block_x + self.kernel_size - 1
                s_height = threads_per_block_y * reg_block_y + self.kernel_size - 1
                shared_mem_size = s_width * s_height * 4  # 4 bytes per float
                
                # Launch kernel
                self.fused_kernel(
                    grid=(blocks_x, blocks_y, blocks_z),
                    block=(threads_per_block_x, threads_per_block_y, 1),
                    args=(cp.asarray(x), cp.asarray(self.weight), cp.asarray(self.bias), 
                         cp.asarray(output), batch_size, in_channels, self.out_channels, 
                         in_height, in_width, self.kernel_size, out_height, out_width),
                    shared_mem=shared_mem_size
                )
                
                return output
                
            except Exception:
                # Fallback to PyTorch implementation if there's an error
                pass
        
        # PyTorch fallback implementation - still optimized with fused bias
        x = F.conv2d(x, self.weight, self.bias)
        return F.mish(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation  
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]