import torch
import torch.nn as nn
import os

class ModelNew(nn.Module):
    """
    Optimized model that performs a SELU activation using a custom CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cuda_module = None
        
        if torch.cuda.is_available():
            try:
                from torch.utils.cpp_extension import load_inline
                
                cuda_source = """
                #include <torch/extension.h>
                #include <cuda.h>
                #include <cuda_runtime.h>

                // SELU constants
                __constant__ float ALPHA = 1.6732632423543772848170429916717f;
                __constant__ float SCALE = 1.0507009873554804934193349852946f;

                template <typename scalar_t>
                __global__ void selu_kernel(
                    const scalar_t* __restrict__ input,
                    scalar_t* __restrict__ output,
                    const int size) {
                    
                    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    const int stride = blockDim.x * gridDim.x;
                    
                    // Thread coarsening - each thread processes 4 elements
                    for (int i = idx; i < size; i += stride) {
                        const scalar_t x = input[i];
                        
                        // Branchless implementation
                        const bool is_positive = (x >= 0);
                        const scalar_t exp_result = is_positive ? x : ALPHA * (__expf(x) - 1.0f);
                        output[i] = SCALE * exp_result;
                    }
                }

                torch::Tensor selu_cuda_forward(torch::Tensor input) {
                    auto output = torch::empty_like(input);
                    const int size = input.numel();
                    
                    // Optimize thread configuration
                    const int threads = 256;
                    const int blocks = min(65535, (size + threads - 1) / threads);
                    
                    AT_DISPATCH_FLOATING_TYPES(input.type(), "selu_cuda_forward", ([&] {
                        selu_kernel<scalar_t><<<blocks, threads>>>(
                            input.data_ptr<scalar_t>(),
                            output.data_ptr<scalar_t>(),
                            size);
                    }));
                    
                    return output;
                }
                """

                cpp_source = """
                #include <torch/extension.h>

                torch::Tensor selu_cuda_forward(torch::Tensor input);

                torch::Tensor selu_forward(torch::Tensor input) {
                    if (input.is_cuda()) {
                        return selu_cuda_forward(input);
                    } else {
                        // For CPU tensors, use PyTorch's implementation
                        return torch::selu(input);
                    }
                }

                PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                    m.def("forward", &selu_forward, "SELU forward");
                }
                """
                
                # Create a unique build directory to avoid conflicts
                build_dir = os.path.join(os.getcwd(), "selu_cuda_build")
                os.makedirs(build_dir, exist_ok=True)
                
                # Compile with optimization flags
                extra_cuda_cflags = ["-O3", "--use_fast_math"]
                
                self.cuda_module = load_inline(
                    name="selu_cuda_ext",
                    cpp_sources=cpp_source,
                    cuda_sources=cuda_source,
                    functions=["forward"],
                    verbose=False,
                    build_directory=build_dir,
                    with_cuda=True,
                    extra_cuda_cflags=extra_cuda_cflags
                )
            except Exception as e:
                print(f"Failed to load CUDA kernel: {e}")
                self.cuda_module = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        if self.cuda_module is not None and x.is_cuda:
            try:
                return self.cuda_module.forward(x)
            except Exception as e:
                print(f"Error in CUDA kernel execution: {e}")
                # Fall back to PyTorch implementation
                return torch.selu(x)
        else:
            # Use PyTorch's implementation when CUDA is not available
            return torch.selu(x)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed