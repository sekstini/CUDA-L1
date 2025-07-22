import torch
import torch.nn as nn
import torch.utils.cpp_extension
import os

class ModelNew(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.
    Optimized implementation using custom CUDA kernel.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cuda_extension_available = False
        
        # Define CUDA kernel for optimized Huber Loss
        cuda_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        #define BLOCK_SIZE 256
        #define ELEMENTS_PER_THREAD 4

        template <typename scalar_t>
        __global__ void huber_loss_kernel(
            const scalar_t* __restrict__ predictions,
            const scalar_t* __restrict__ targets,
            scalar_t* __restrict__ output,
            const int64_t size) {
            
            __shared__ scalar_t shared_mem[BLOCK_SIZE];
            
            const int tid = threadIdx.x;
            const int bid = blockIdx.x;
            const int blockSize = blockDim.x;
            const int gridSize = gridDim.x;
            
            // Calculate base index for this thread
            int base_idx = (bid * blockSize + tid) * ELEMENTS_PER_THREAD;
            const int stride = blockSize * gridSize * ELEMENTS_PER_THREAD;
            
            // Initialize thread accumulator
            scalar_t thread_sum = 0.0f;
            
            // Process multiple elements per thread in a grid-stride loop
            while (base_idx < size) {
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const int idx = base_idx + i;
                    if (idx < size) {
                        const scalar_t diff = predictions[idx] - targets[idx];
                        const scalar_t abs_diff = fabsf(diff);
                        
                        // PyTorch smooth_l1_loss implementation (beta=1.0)
                        thread_sum += (abs_diff < 1.0f) ? 
                                    (0.5f * diff * diff) : 
                                    (abs_diff - 0.5f);
                    }
                }
                base_idx += stride;
            }
            
            // Store thread sum to shared memory
            shared_mem[tid] = thread_sum;
            __syncthreads();
            
            // Parallel reduction in shared memory
            for (int s = blockSize / 2; s > 32; s >>= 1) {
                if (tid < s) {
                    shared_mem[tid] += shared_mem[tid + s];
                }
                __syncthreads();
            }
            
            // Warp-level reduction (no sync needed within a warp)
            if (tid < 32) {
                // Use warp shuffle operations for the final reduction steps
                scalar_t val = shared_mem[tid];
                
                // Unroll the final warp reduction completely
                if (blockSize >= 64) val += shared_mem[tid + 32];
                __syncwarp();
                
                val += __shfl_down_sync(0xffffffff, val, 16);
                val += __shfl_down_sync(0xffffffff, val, 8);
                val += __shfl_down_sync(0xffffffff, val, 4);
                val += __shfl_down_sync(0xffffffff, val, 2);
                val += __shfl_down_sync(0xffffffff, val, 1);
                
                // First thread in each block adds its result to the output using atomics
                if (tid == 0) {
                    atomicAdd(output, val);
                }
            }
        }

        template <typename scalar_t>
        __global__ void finalize_kernel(scalar_t* output, const int64_t size) {
            // Divide by size to get the mean
            output[0] = output[0] / static_cast<scalar_t>(size);
        }

        torch::Tensor huber_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
            // Ensure inputs are contiguous for optimal memory access
            predictions = predictions.contiguous();
            targets = targets.contiguous();
            
            const int64_t size = predictions.numel();
            auto output = torch::zeros({1}, predictions.options());
            
            // Configure kernel launch parameters
            const int threads = BLOCK_SIZE;
            
            // Calculate optimal grid size based on input dimensions and elements per thread
            const int blocks = min(1024, (size + threads * ELEMENTS_PER_THREAD - 1) / (threads * ELEMENTS_PER_THREAD));
            
            AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "huber_loss_cuda", ([&] {
                // Launch kernel to compute the sum
                huber_loss_kernel<scalar_t><<<blocks, threads>>>(
                    predictions.data_ptr<scalar_t>(),
                    targets.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    size);
                
                // Launch kernel to compute the mean
                finalize_kernel<scalar_t><<<1, 1>>>(
                    output.data_ptr<scalar_t>(),
                    size);
            }));
            
            return output;
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("forward", &huber_loss_cuda, "Huber Loss forward (CUDA)");
        }
        """
        
        # Try to compile the CUDA extension
        try:
            self.huber_loss_cuda = torch.utils.cpp_extension.load_inline(
                name="huber_loss_cuda",
                cpp_sources="",
                cuda_sources=cuda_source,
                functions=["forward"],
                with_cuda=True,
                build_directory=os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions"),
                verbose=False
            )
            self.cuda_extension_available = True
        except Exception as e:
            print(f"Failed to load CUDA extension: {e}")
            self.cuda_extension_available = False
    
    def forward(self, predictions, targets):
        # Use custom CUDA kernel if available and inputs are on GPU
        if (self.cuda_extension_available and 
            predictions.is_cuda and 
            targets.is_cuda and
            predictions.dtype == targets.dtype):
            return self.huber_loss_cuda.forward(predictions, targets)
        else:
            # Fall back to PyTorch implementation
            return torch.nn.functional.smooth_l1_loss(predictions, targets)

# Keep hyperparameters exactly as in reference implementation
batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return []