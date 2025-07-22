import torch
import torch.nn as nn
import torch.utils.cpp_extension
import os

class ModelNew(nn.Module):
    """
    Optimized model that performs a LogSoftmax activation.
    
    Args:
        dim (int): The dimension along which LogSoftmax will be computed.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        
        # Define CUDA kernel for LogSoftmax
        cuda_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <vector>

        template <typename scalar_t>
        __device__ __forceinline__ scalar_t warp_reduce_max(scalar_t val) {
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
            return val;
        }

        template <typename scalar_t>
        __device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                val += __shfl_xor_sync(0xffffffff, val, offset);
            return val;
        }

        template <typename scalar_t>
        __global__ void log_softmax_kernel(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ output,
            const int batch_size,
            const int dim) {
            
            // Each block handles one row (one sample in the batch)
            const int batch_idx = blockIdx.x;
            if (batch_idx >= batch_size) return;
            
            // Get pointers to current row
            const scalar_t* row_input = input + batch_idx * dim;
            scalar_t* row_output = output + batch_idx * dim;
            
            // Shared memory for reductions
            extern __shared__ char shared_mem[];
            scalar_t* s_data = reinterpret_cast<scalar_t*>(shared_mem);
            
            const int tid = threadIdx.x;
            const int lane_id = tid % 32;
            const int warp_id = tid / 32;
            const int warps_per_block = blockDim.x / 32;
            
            // Find max value in this row
            scalar_t thread_max = -INFINITY;
            
            // Process elements with grid-stride loop for better memory coalescing
            #pragma unroll 4
            for (int i = tid; i < dim; i += blockDim.x) {
                thread_max = max(thread_max, row_input[i]);
            }
            
            // Warp-level reduction to find max
            thread_max = warp_reduce_max(thread_max);
            
            // Store the warp results in shared memory
            if (lane_id == 0) {
                s_data[warp_id] = thread_max;
            }
            __syncthreads();
            
            // Final reduction across warps
            scalar_t row_max = -INFINITY;
            if (tid < warps_per_block) {
                row_max = s_data[tid];
                row_max = warp_reduce_max(row_max);
            }
            
            // Broadcast max to all threads
            if (tid == 0) {
                s_data[0] = row_max;
            }
            __syncthreads();
            row_max = s_data[0];
            
            // Compute sum of exp(x - max)
            scalar_t thread_sum = 0;
            
            // Process elements with grid-stride loop
            #pragma unroll 4
            for (int i = tid; i < dim; i += blockDim.x) {
                thread_sum += __expf(row_input[i] - row_max);
            }
            
            // Warp-level reduction to find sum
            thread_sum = warp_reduce_sum(thread_sum);
            
            // Store the warp results in shared memory
            if (lane_id == 0) {
                s_data[warp_id] = thread_sum;
            }
            __syncthreads();
            
            // Final reduction across warps
            scalar_t sum = 0;
            if (tid < warps_per_block) {
                sum = s_data[tid];
                sum = warp_reduce_sum(sum);
            }
            
            // Broadcast sum to all threads
            if (tid == 0) {
                s_data[0] = sum;
            }
            __syncthreads();
            sum = s_data[0];
            
            // Compute log(sum) once
            scalar_t log_sum = __logf(sum);
            
            // Compute final output: x - max - log(sum(exp(x - max)))
            #pragma unroll 4
            for (int i = tid; i < dim; i += blockDim.x) {
                row_output[i] = row_input[i] - row_max - log_sum;
            }
        }

        // Specialized kernel for float using float4 vectorized loads/stores
        __global__ void log_softmax_kernel_float4(
            const float* __restrict__ input,
            float* __restrict__ output,
            const int batch_size,
            const int dim) {
            
            // Each block handles one row (one sample in the batch)
            const int batch_idx = blockIdx.x;
            if (batch_idx >= batch_size) return;
            
            // Get pointers to current row
            const float* row_input = input + batch_idx * dim;
            float* row_output = output + batch_idx * dim;
            
            // Shared memory for reductions
            extern __shared__ char shared_mem[];
            float* s_data = reinterpret_cast<float*>(shared_mem);
            
            const int tid = threadIdx.x;
            const int lane_id = tid % 32;
            const int warp_id = tid / 32;
            const int warps_per_block = blockDim.x / 32;
            
            // Find max value in this row
            float thread_max = -INFINITY;
            
            // Process elements with grid-stride loop using float4 when possible
            const float4* row_input4 = reinterpret_cast<const float4*>(row_input);
            const int dim4 = dim / 4;
            
            #pragma unroll 2
            for (int i = tid; i < dim4; i += blockDim.x) {
                float4 values = row_input4[i];
                thread_max = max(thread_max, values.x);
                thread_max = max(thread_max, values.y);
                thread_max = max(thread_max, values.z);
                thread_max = max(thread_max, values.w);
            }
            
            // Warp-level reduction to find max
            thread_max = warp_reduce_max(thread_max);
            
            // Store the warp results in shared memory
            if (lane_id == 0) {
                s_data[warp_id] = thread_max;
            }
            __syncthreads();
            
            // Final reduction across warps
            float row_max = -INFINITY;
            if (tid < warps_per_block) {
                row_max = s_data[tid];
                row_max = warp_reduce_max(row_max);
            }
            
            // Broadcast max to all threads
            if (tid == 0) {
                s_data[0] = row_max;
            }
            __syncthreads();
            row_max = s_data[0];
            
            // Compute sum of exp(x - max)
            float thread_sum = 0;
            
            // Process elements with grid-stride loop using float4
            #pragma unroll 2
            for (int i = tid; i < dim4; i += blockDim.x) {
                float4 values = row_input4[i];
                thread_sum += __expf(values.x - row_max);
                thread_sum += __expf(values.y - row_max);
                thread_sum += __expf(values.z - row_max);
                thread_sum += __expf(values.w - row_max);
            }
            
            // Warp-level reduction to find sum
            thread_sum = warp_reduce_sum(thread_sum);
            
            // Store the warp results in shared memory
            if (lane_id == 0) {
                s_data[warp_id] = thread_sum;
            }
            __syncthreads();
            
            // Final reduction across warps
            float sum = 0;
            if (tid < warps_per_block) {
                sum = s_data[tid];
                sum = warp_reduce_sum(sum);
            }
            
            // Broadcast sum to all threads
            if (tid == 0) {
                s_data[0] = sum;
            }
            __syncthreads();
            sum = s_data[0];
            
            // Compute log(sum) once
            float log_sum = __logf(sum);
            
            // Compute final output: x - max - log(sum(exp(x - max)))
            float4* row_output4 = reinterpret_cast<float4*>(row_output);
            
            #pragma unroll 2
            for (int i = tid; i < dim4; i += blockDim.x) {
                float4 values = row_input4[i];
                float4 result;
                result.x = values.x - row_max - log_sum;
                result.y = values.y - row_max - log_sum;
                result.z = values.z - row_max - log_sum;
                result.w = values.w - row_max - log_sum;
                row_output4[i] = result;
            }
        }

        std::vector<torch::Tensor> log_softmax_cuda_forward(
            torch::Tensor input,
            int dim) {
            
            // Check input
            TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
            TORCH_CHECK(dim >= 0 && dim < input.dim(), "Invalid dimension");
            TORCH_CHECK(dim == 1, "Custom CUDA kernel only supports dim=1");
            TORCH_CHECK(input.dim() == 2, "Custom CUDA kernel only supports 2D tensors");
            
            // Get tensor dimensions
            const int batch_size = input.size(0);
            const int feature_dim = input.size(1);
            
            // Create output tensor
            auto output = torch::empty_like(input);
            
            // Calculate optimal thread count
            const int threads = 256;  // Good balance for most GPUs
            
            // Calculate shared memory size for reductions
            const int warps_per_block = threads / 32;
            const size_t shared_mem_size = sizeof(float) * warps_per_block;
            
            // Choose the appropriate kernel based on data type and alignment
            if (input.scalar_type() == torch::ScalarType::Float && 
                feature_dim % 4 == 0 && 
                input.is_contiguous() && 
                output.is_contiguous()) {
                // Use vectorized kernel for float tensors with dimension divisible by 4
                const float* input_ptr = input.data_ptr<float>();
                float* output_ptr = output.data_ptr<float>();
                
                log_softmax_kernel_float4<<<batch_size, threads, shared_mem_size>>>(
                    input_ptr, output_ptr, batch_size, feature_dim);
            } else {
                // Use standard kernel for other cases
                AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda", ([&] {
                    const scalar_t* input_ptr = input.data_ptr<scalar_t>();
                    scalar_t* output_ptr = output.data_ptr<scalar_t>();
                    
                    log_softmax_kernel<scalar_t><<<batch_size, threads, shared_mem_size>>>(
                        input_ptr, output_ptr, batch_size, feature_dim);
                }));
            }
            
            return {output};
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward (CUDA)");
        }
        """
        
        # Try to load the extension, with fallback to PyTorch implementation
        self.cuda_extension_available = False
        if torch.cuda.is_available():
            try:
                # Create a directory for storing compiled extensions if it doesn't exist
                os.makedirs('cuda_extensions', exist_ok=True)
                
                # Load the CUDA extension
                self.log_softmax_cuda = torch.utils.cpp_extension.load_inline(
                    name='log_softmax_cuda',
                    cpp_sources='',
                    cuda_sources=cuda_source,
                    functions=['forward'],
                    with_cuda=True,
                    build_directory='cuda_extensions',
                    verbose=False,
                    extra_cuda_cflags=['-O3', '--use_fast_math']
                )
                
                self.cuda_extension_available = True
            except Exception as e:
                print(f"Warning: Could not load CUDA extension: {e}")
                print("Falling back to PyTorch implementation")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        # Use our custom CUDA implementation if available and applicable
        if self.cuda_extension_available and x.is_cuda and self.dim == 1 and x.dim() == 2:
            try:
                # Make sure input is contiguous
                if not x.is_contiguous():
                    x = x.contiguous()
                
                # Forward pass through custom CUDA kernel
                return self.log_softmax_cuda.forward(x, self.dim)[0]
            except Exception as e:
                # Fall back to PyTorch implementation if there's an error
                print(f"Warning: Custom CUDA kernel failed with error: {e}")
                print("Falling back to PyTorch implementation")
                return torch.log_softmax(x, self.dim)
        else:
            # Fall back to PyTorch implementation
            return torch.log_softmax(x, self.dim)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(batch_size, dim, device=device)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed