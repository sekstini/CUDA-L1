import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

class ModelNew(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.
    Optimized with custom CUDA kernel.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """
    
    _cuda_module = None
    
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        
        # Load CUDA extension if not already loaded
        if ModelNew._cuda_module is None:
            try:
                cuda_source = '''
                #include <torch/extension.h>
                #include <cuda.h>
                #include <cuda_runtime.h>

                template <typename scalar_t>
                __global__ void masked_cumsum_kernel(
                    const scalar_t* __restrict__ input,
                    const bool* __restrict__ mask,
                    scalar_t* __restrict__ output,
                    const int seq_length) 
                {
                    // Each block processes one batch element
                    const int batch_idx = blockIdx.x;
                    const int tid = threadIdx.x;
                    
                    // Calculate offsets for this batch element
                    const int batch_offset = batch_idx * seq_length;
                    const scalar_t* batch_input = input + batch_offset;
                    const bool* batch_mask = mask + batch_offset;
                    scalar_t* batch_output = output + batch_offset;
                    
                    // Shared memory for efficient scan
                    extern __shared__ scalar_t shared_mem[];
                    
                    // Process the sequence in chunks if needed
                    scalar_t running_sum = 0.0f;
                    
                    for (int chunk_start = 0; chunk_start < seq_length; chunk_start += blockDim.x) {
                        const int idx = chunk_start + tid;
                        
                        // Load data into shared memory
                        if (idx < seq_length) {
                            shared_mem[tid] = batch_mask[idx] ? batch_input[idx] : 0.0f;
                        } else {
                            shared_mem[tid] = 0.0f;
                        }
                        
                        __syncthreads();
                        
                        // Perform Hillis-Steele scan in shared memory
                        for (int stride = 1; stride < blockDim.x; stride *= 2) {
                            scalar_t val = 0.0f;
                            if (tid >= stride) {
                                val = shared_mem[tid - stride];
                            }
                            __syncthreads();
                            
                            if (tid >= stride) {
                                shared_mem[tid] += val;
                            }
                            __syncthreads();
                        }
                        
                        // Add running sum from previous chunks
                        if (tid < blockDim.x) {
                            shared_mem[tid] += running_sum;
                        }
                        __syncthreads();
                        
                        // Write results to output
                        if (idx < seq_length) {
                            batch_output[idx] = shared_mem[tid];
                        }
                        
                        // Update running sum for next chunk
                        if (blockDim.x > 0) {
                            running_sum = shared_mem[blockDim.x - 1];
                        }
                        
                        __syncthreads();
                    }
                }

                torch::Tensor masked_cumsum_cuda(
                    torch::Tensor input,
                    torch::Tensor mask,
                    int dim) 
                {
                    TORCH_CHECK(dim == 1, "Only dim=1 is currently supported");
                    
                    const auto batch_size = input.size(0);
                    const auto seq_length = input.size(1);
                    
                    auto output = torch::zeros_like(input);
                    
                    // Optimize block size based on sequence length
                    const int block_size = 256;  // Good balance for most GPUs
                    const int grid_size = batch_size;
                    const int shared_mem_size = block_size * sizeof(float);
                    
                    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "masked_cumsum_cuda", ([&] {
                        masked_cumsum_kernel<scalar_t><<<grid_size, block_size, shared_mem_size>>>(
                            input.data_ptr<scalar_t>(),
                            mask.data_ptr<bool>(),
                            output.data_ptr<scalar_t>(),
                            seq_length
                        );
                    }));
                    
                    return output;
                }
                '''

                cpp_source = '''
                #include <torch/extension.h>

                torch::Tensor masked_cumsum_cuda(
                    torch::Tensor input,
                    torch::Tensor mask,
                    int dim);

                torch::Tensor masked_cumsum(
                    torch::Tensor input,
                    torch::Tensor mask,
                    int dim) 
                {
                    if (dim != 1 || !input.is_cuda()) {
                        return torch::cumsum(input * mask, dim);
                    }
                    
                    return masked_cumsum_cuda(input.contiguous(), mask.contiguous(), dim);
                }

                PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                    m.def("masked_cumsum", &masked_cumsum, "Masked cumulative sum");
                }
                '''
                
                # Create a unique module name to avoid conflicts
                module_name = f"masked_cumsum_{os.getpid()}"
                
                # Load the CUDA extension
                ModelNew._cuda_module = load_inline(
                    name=module_name,
                    cpp_sources=cpp_source,
                    cuda_sources=cuda_source,
                    functions=["masked_cumsum"],
                    verbose=False
                )
            except Exception as e:
                print(f"Failed to load CUDA extension: {e}")
                ModelNew._cuda_module = None

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        # Fall back to PyTorch implementation if CUDA extension failed to load
        if ModelNew._cuda_module is None or self.dim != 1 or not x.is_cuda:
            return torch.cumsum(x * mask, dim=self.dim)
        
        # Make sure inputs are contiguous
        x = x.contiguous()
        mask = mask.contiguous()
        
        try:
            # Use our custom CUDA kernel
            return ModelNew._cuda_module.masked_cumsum(x, mask, self.dim)
        except Exception as e:
            # Fall back to PyTorch implementation if CUDA kernel fails
            print(f"CUDA kernel failed: {e}")
            return torch.cumsum(x * mask, dim=self.dim)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    x = torch.randn(batch_size, *input_shape)
    mask = torch.randint(0, 2, x.shape).bool()  # Random boolean mask
    return [x, mask]

def get_init_inputs():
    return [dim]