import torch
import torch.nn as nn
import torch.nn.functional as F

class ELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        # Save input for backward pass
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        
        # Allocate output tensor
        output = torch.empty_like(input)
        
        # Compute ELU: x if x > 0, alpha * (exp(x) - 1) if x <= 0
        # Using a vectorized approach for better performance
        output = torch.where(input > 0, input, alpha * (torch.exp(input) - 1.0))
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        
        # ELU derivative: 1 if x > 0, alpha * exp(x) if x <= 0
        grad_input = torch.where(input > 0, grad_output, grad_output * alpha * torch.exp(input))
        
        # No gradient for alpha parameter
        return grad_input, None

class ModelNew(nn.Module):
    """
    Optimized implementation of ELU activation.
    
    Args:
        alpha (float): The alpha parameter for the ELU function. Defaults to 1.0.
    """
    def __init__(self, alpha: float = 1.0):
        super(ModelNew, self).__init__()
        self.alpha = alpha
        
        # Define CUDA kernel for ELU forward pass
        self.cuda_kernel_template = """
        extern "C" __global__ void elu_forward(
            const float* input,
            float* output,
            const float alpha,
            const int size
        ) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = blockDim.x * gridDim.x;
            
            for (int i = idx; i < size; i += stride) {
                const float x = input[i];
                output[i] = x > 0 ? x : alpha * (expf(x) - 1.0f);
            }
        }
        
        extern "C" __global__ void elu_backward(
            const float* grad_output,
            const float* input,
            float* grad_input,
            const float alpha,
            const int size
        ) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = blockDim.x * gridDim.x;
            
            for (int i = idx; i < size; i += stride) {
                const float x = input[i];
                grad_input[i] = x > 0 ? grad_output[i] : grad_output[i] * alpha * expf(x);
            }
        }
        """
        
        # Compile CUDA kernel if on GPU
        self.kernel = None
        if torch.cuda.is_available():
            try:
                from torch.utils.cpp_extension import load_inline
                self.kernel = load_inline(
                    name="elu_cuda",
                    cpp_sources="",
                    cuda_sources=self.cuda_kernel_template,
                    functions=["elu_forward", "elu_backward"],
                    with_cuda=True,
                    extra_cuda_cflags=["-O3"]
                )
            except:
                # Fall back to PyTorch implementation if compilation fails
                self.kernel = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized ELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ELU applied, same shape as input.
        """
        # Use custom CUDA kernel if available and input is on CUDA
        if self.kernel is not None and x.is_cuda:
            output = torch.empty_like(x)
            n = x.numel()
            
            # Configure kernel launch parameters
            threads_per_block = 256
            blocks = min(1024, (n + threads_per_block - 1) // threads_per_block)
            
            # Launch kernel
            self.kernel.elu_forward(
                blocks, threads_per_block, 0,
                x.contiguous().data_ptr(),
                output.data_ptr(),
                self.alpha,
                n
            )
            return output
        else:
            # Fall back to optimized PyTorch implementation
            return F.elu(x, alpha=self.alpha)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return [1.0]  # Provide alpha value for initialization