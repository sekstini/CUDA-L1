import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution, adds a value,
    takes the minimum, applies GELU, and multiplies by a value.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        stride (int): Stride of the convolution
        add_value (float): Value to add after convolution
        multiply_value (float): Value to multiply after GELU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        
        # Enable cuDNN autotuner for faster convolutions
        cudnn.benchmark = True
        
        # Pre-convert weights to channels_last format if possible
        if torch.cuda.is_available():
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(
                memory_format=torch.channels_last)
        
        # Initialize CUDA kernel for fused post-processing
        self.fused_kernel = None
        if torch.cuda.is_available():
            self._init_cuda_kernel()
    
    def _init_cuda_kernel(self):
        """Initialize the CUDA kernel for fused post-processing operations"""
        cuda_code = """
        #include <cuda_runtime.h>
        
        // CUDA kernel for fused post-processing operations (scalar version)
        extern "C" __global__ void fused_post_process(
            float* __restrict__ output,
            const float* __restrict__ input,
            const int numel,
            const float add_value,
            const float multiply_value)
        {
            // Calculate global thread position
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Constants for GELU approximation
            const float sqrt_2_over_pi = 0.7978845608028654f;
            const float coef = 0.044715f;
            
            // Each thread processes multiple elements for better efficiency
            const int stride = blockDim.x * gridDim.x;
            
            for (int i = idx; i < numel; i += stride) {
                // Load input value
                float val = input[i];
                
                // Add operation
                val = val + add_value;
                
                // Min operation
                val = min(val, 0.0f);
                
                // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                float val_cubed = val * val * val;
                float inner = sqrt_2_over_pi * (val + coef * val_cubed);
                float tanh_inner = tanhf(inner);
                val = 0.5f * val * (1.0f + tanh_inner);
                
                // Multiply operation
                val = val * multiply_value;
                
                // Store result
                output[i] = val;
            }
        }
        
        // Vectorized version for better memory throughput when aligned
        extern "C" __global__ void fused_post_process_vec4(
            float4* __restrict__ output,
            const float4* __restrict__ input,
            const int vec_numel,
            const float add_value,
            const float multiply_value)
        {
            // Calculate global thread position
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (idx >= vec_numel) return;
            
            // Constants for GELU approximation
            const float sqrt_2_over_pi = 0.7978845608028654f;
            const float coef = 0.044715f;
            
            // Load input vector (4 values at once)
            float4 val4 = input[idx];
            
            // Process each component of the vector
            
            // Component 1
            float val = val4.x;
            val = val + add_value;
            val = min(val, 0.0f);
            float val_cubed = val * val * val;
            float inner = sqrt_2_over_pi * (val + coef * val_cubed);
            float tanh_inner = tanhf(inner);
            val = 0.5f * val * (1.0f + tanh_inner);
            val4.x = val * multiply_value;
            
            // Component 2
            val = val4.y;
            val = val + add_value;
            val = min(val, 0.0f);
            val_cubed = val * val * val;
            inner = sqrt_2_over_pi * (val + coef * val_cubed);
            tanh_inner = tanhf(inner);
            val = 0.5f * val * (1.0f + tanh_inner);
            val4.y = val * multiply_value;
            
            // Component 3
            val = val4.z;
            val = val + add_value;
            val = min(val, 0.0f);
            val_cubed = val * val * val;
            inner = sqrt_2_over_pi * (val + coef * val_cubed);
            tanh_inner = tanhf(inner);
            val = 0.5f * val * (1.0f + tanh_inner);
            val4.z = val * multiply_value;
            
            // Component 4
            val = val4.w;
            val = val + add_value;
            val = min(val, 0.0f);
            val_cubed = val * val * val;
            inner = sqrt_2_over_pi * (val + coef * val_cubed);
            tanh_inner = tanhf(inner);
            val = 0.5f * val * (1.0f + tanh_inner);
            val4.w = val * multiply_value;
            
            // Store result vector (4 values at once)
            output[idx] = val4;
        }
        """
        
        from torch.utils.cpp_extension import load_inline
        
        try:
            self.fused_kernel = load_inline(
                name="fused_post_process",
                cpp_sources="""
                #include <torch/extension.h>
                #include <cuda_runtime.h>
                
                // Forward declaration of the CUDA kernels
                extern "C" __global__ void fused_post_process(
                    float* output,
                    const float* input,
                    const int numel,
                    const float add_value,
                    const float multiply_value);
                    
                extern "C" __global__ void fused_post_process_vec4(
                    float4* output,
                    const float4* input,
                    const int vec_numel,
                    const float add_value,
                    const float multiply_value);
                
                torch::Tensor fused_post_process_cuda(
                    torch::Tensor input,
                    float add_value,
                    float multiply_value)
                {
                    // Check input tensor
                    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
                    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
                    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
                    
                    // Get tensor size
                    int numel = input.numel();
                    
                    // Create output tensor
                    auto output = torch::empty_like(input);
                    
                    // Check if we can use vectorized version (numel divisible by 4 and address alignment)
                    bool use_vec4 = (numel % 4 == 0) && 
                                   (reinterpret_cast<uintptr_t>(input.data_ptr<float>()) % 16 == 0) &&
                                   (reinterpret_cast<uintptr_t>(output.data_ptr<float>()) % 16 == 0);
                    
                    // Configure kernel launch parameters
                    const int threads = 256;
                    
                    if (use_vec4) {
                        const int vec_numel = numel / 4;
                        const int blocks = min(65535, (vec_numel + threads - 1) / threads);
                        
                        // Launch vectorized kernel
                        fused_post_process_vec4<<<blocks, threads>>>(
                            reinterpret_cast<float4*>(output.data_ptr<float>()),
                            reinterpret_cast<const float4*>(input.data_ptr<float>()),
                            vec_numel,
                            add_value,
                            multiply_value
                        );
                    } else {
                        const int blocks = min(65535, (numel + threads - 1) / threads);
                        
                        // Launch standard kernel
                        fused_post_process<<<blocks, threads>>>(
                            output.data_ptr<float>(),
                            input.data_ptr<float>(),
                            numel,
                            add_value,
                            multiply_value
                        );
                    }
                    
                    // Check for CUDA errors
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        throw std::runtime_error(cudaGetErrorString(err));
                    }
                    
                    return output;
                }
                
                PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                    m.def("fused_post_process_cuda", &fused_post_process_cuda, "Fused post-processing operations");
                }
                """,
                cuda_sources=cuda_code,
                functions=["fused_post_process_cuda"],
                verbose=False
            )
        except Exception as e:
            print(f"Failed to load CUDA kernel: {e}")
            self.fused_kernel = None
    
    def forward(self, x):
        # Check if we can use channels_last memory format
        use_channels_last = (x.is_cuda and x.dim() == 4 and 
                            x.size(0) >= 8 and x.size(2) >= 8 and x.size(3) >= 8)
        
        if use_channels_last:
            # Convert to channels_last memory format for potentially better performance
            x = x.to(memory_format=torch.channels_last)
            
            # Perform the transposed convolution with channels_last tensors
            if self.conv_transpose.bias is not None:
                bias = self.conv_transpose.bias
            else:
                bias = None
                
            # Use F.conv_transpose2d directly with channels_last tensors
            x = torch.nn.functional.conv_transpose2d(
                x, self.conv_transpose.weight, bias,
                stride=self.conv_transpose.stride,
                padding=self.conv_transpose.padding,
                output_padding=self.conv_transpose.output_padding,
                groups=self.conv_transpose.groups,
                dilation=self.conv_transpose.dilation
            )
        else:
            # Use standard convolution
            x = self.conv_transpose(x)
        
        # Apply fused post-processing if CUDA kernel is available
        if self.fused_kernel is not None and x.is_cuda:
            try:
                # Ensure tensor is contiguous in the current memory format
                x = x.contiguous()
                return self.fused_kernel.fused_post_process_cuda(x, self.add_value, self.multiply_value)
            except Exception as e:
                # Fall back to standard implementation if CUDA kernel fails
                pass
        
        # Standard implementation as fallback
        x = x + self.add_value
        x = torch.min(x, torch.tensor(0.0, device=x.device))
        x = torch.nn.functional.gelu(x)
        x = x * self.multiply_value
        return x

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 32
out_channels = 16
height, width = 32, 32
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]