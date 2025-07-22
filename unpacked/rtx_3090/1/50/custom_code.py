import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

class ModelNew(nn.Module):
    """
    Optimized model that performs product reduction over a dimension.
    
    Args:
        dim (int): Dimension to reduce over.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cuda_module = None
        
        # Only compile the CUDA extension if CUDA is available
        if torch.cuda.is_available():
            try:
                # CUDA kernel for product reduction using logarithm technique
                cuda_source = '''
                #include <torch/extension.h>
                #include <cuda.h>
                #include <cuda_runtime.h>
                #include <math.h>
                
                template <typename scalar_t>
                __device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        val += __shfl_down_sync(0xffffffff, val, offset);
                    }
                    return val;
                }
                
                template <typename scalar_t>
                __device__ __forceinline__ int warpReduceOr(int val) {
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        val |= __shfl_down_sync(0xffffffff, val, offset);
                    }
                    return val;
                }
                
                template <typename scalar_t>
                __device__ __forceinline__ int warpReduceXor(int val) {
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        val ^= __shfl_down_sync(0xffffffff, val, offset);
                    }
                    return val;
                }
                
                template <typename scalar_t>
                __global__ void product_reduction_kernel(
                    const scalar_t* __restrict__ input,
                    scalar_t* __restrict__ output,
                    const int batch_size,
                    const int dim1,
                    const int dim2) {
                    
                    // Calculate indices
                    const int b = blockIdx.x;  // batch index
                    const int d = blockIdx.y;  // output dimension index
                    
                    // Check bounds
                    if (b >= batch_size || d >= dim2) return;
                    
                    // Shared memory for partial results from each warp
                    __shared__ scalar_t warp_sums[8];  // For 256 threads = 8 warps
                    __shared__ int warp_signs[8];
                    __shared__ int warp_has_zero[8];
                    
                    const int tid = threadIdx.x;
                    const int warpId = tid / 32;
                    const int laneId = tid % 32;
                    
                    // Initialize values
                    scalar_t thread_sum = 0.0;
                    int thread_sign_odd = 0;  // 0 = even number of negatives, 1 = odd number of negatives
                    int thread_has_zero = 0;
                    
                    // Each thread processes elements with stride for better memory coalescing
                    const int input_offset = b * dim1 * dim2 + d;
                    
                    // Process multiple elements per thread with striding
                    for (int i = tid; i < dim1; i += blockDim.x) {
                        scalar_t val = input[input_offset + i * dim2];
                        
                        if (val == 0.0) {
                            thread_has_zero = 1;
                            break;
                        } else if (val < 0.0) {
                            thread_sign_odd ^= 1;  // Toggle sign (XOR with 1)
                            val = -val;  // Take absolute value
                        }
                        
                        // Use log for numerical stability
                        thread_sum += log(val);
                    }
                    
                    // Warp-level reduction
                    int warp_zero = warpReduceOr(thread_has_zero);
                    int warp_sign_odd = warpReduceXor(thread_sign_odd);
                    scalar_t warp_sum = warpReduceSum(thread_sum);
                    
                    // First thread in each warp writes to shared memory
                    if (laneId == 0) {
                        warp_sums[warpId] = warp_sum;
                        warp_signs[warpId] = warp_sign_odd;
                        warp_has_zero[warpId] = warp_zero;
                    }
                    
                    __syncthreads();
                    
                    // Final reduction using first warp
                    if (warpId == 0) {
                        scalar_t block_sum = 0.0;
                        int block_sign_odd = 0;
                        int block_has_zero = 0;
                        
                        if (laneId < 8) {  // Only first 8 lanes (one per warp)
                            block_sum = warp_sums[laneId];
                            block_sign_odd = warp_signs[laneId];
                            block_has_zero = warp_has_zero[laneId];
                        }
                        
                        // Final warp reduction
                        block_has_zero = warpReduceOr(block_has_zero);
                        
                        // Only compute the final result if there are no zeros
                        if (!block_has_zero) {
                            block_sign_odd = warpReduceXor(block_sign_odd);
                            block_sum = warpReduceSum(block_sum);
                        }
                        
                        // Write final result
                        if (laneId == 0) {
                            if (block_has_zero) {
                                output[b * dim2 + d] = 0.0;
                            } else {
                                scalar_t result = exp(block_sum);
                                output[b * dim2 + d] = block_sign_odd ? -result : result;
                            }
                        }
                    }
                }
                
                torch::Tensor product_reduction_cuda(torch::Tensor input, int dim) {
                    // Validate input
                    TORCH_CHECK(input.dim() == 3, "Input must be 3D tensor");
                    TORCH_CHECK(dim == 1, "Only reduction along dimension 1 is currently supported");
                    
                    // Get tensor dimensions
                    const auto batch_size = input.size(0);
                    const auto dim1 = input.size(1);
                    const auto dim2 = input.size(2);
                    
                    // Create output tensor
                    auto output = torch::empty({batch_size, dim2}, input.options());
                    
                    // Calculate kernel launch parameters
                    const int threads_per_block = 256;
                    const dim3 blocks(batch_size, dim2);
                    
                    // Launch kernel
                    AT_DISPATCH_FLOATING_TYPES(input.type(), "product_reduction_cuda", ([&] {
                        product_reduction_kernel<scalar_t><<<blocks, threads_per_block, 0>>>(
                            input.data_ptr<scalar_t>(),
                            output.data_ptr<scalar_t>(),
                            batch_size,
                            dim1,
                            dim2
                        );
                    }));
                    
                    // Check for kernel launch errors
                    cudaError_t err = cudaGetLastError();
                    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
                    
                    return output;
                }
                '''
                
                cpp_source = '''
                #include <torch/extension.h>
                
                torch::Tensor product_reduction_cuda(torch::Tensor input, int dim);
                
                torch::Tensor product_reduction(torch::Tensor input, int dim) {
                    if (input.device().is_cuda() && dim == 1 && input.dim() == 3) {
                        // Ensure input is contiguous for better performance
                        auto x_cont = input.contiguous();
                        return product_reduction_cuda(x_cont, dim);
                    } else {
                        // Fallback to PyTorch implementation
                        return torch::prod(input, dim);
                    }
                }
                
                PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                    m.def("product_reduction", &product_reduction, "Optimized product reduction operation");
                }
                '''
                
                # Create a temporary directory for the extension
                os.makedirs("tmp_product_opt", exist_ok=True)
                
                # Load the custom CUDA kernel
                self.cuda_module = load_inline(
                    name="product_cuda_opt",
                    cpp_sources=[cpp_source],
                    cuda_sources=[cuda_source],
                    functions=["product_reduction"],
                    with_cuda=True,
                    build_directory="tmp_product_opt",
                    verbose=False,
                    extra_cflags=['-O3'],
                    extra_cuda_cflags=['-O3', '--use_fast_math']
                )
            except Exception as e:
                print(f"Failed to load CUDA extension: {e}")
                self.cuda_module = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs product reduction over the specified dimension.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor with product reduction applied.
        """
        # Use custom CUDA kernel if available and applicable
        if (self.cuda_module is not None and x.is_cuda and 
            self.dim == 1 and x.dim() == 3):
            try:
                # Ensure input is contiguous for better performance
                x_cont = x.contiguous() if not x.is_contiguous() else x
                return self.cuda_module.product_reduction(x_cont, self.dim)
            except Exception:
                # Fallback to PyTorch implementation if there's an error
                return torch.prod(x, dim=self.dim)
        else:
            # Use PyTorch's implementation
            return torch.prod(x, dim=self.dim)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim1 = 256
dim2 = 256
reduction_dim = 1

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [reduction_dim]