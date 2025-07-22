import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

# Custom CUDA kernel for 1D max pooling
kernel_code = '''
extern "C"
__global__ void max_pool1d_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {
    
    // Calculate indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // Each thread processes multiple elements
    for (int linear_idx = idx; linear_idx < batch_size * channels * output_length; linear_idx += total_threads) {
        // Convert linear index to 3D coordinates
        int b = linear_idx / (channels * output_length);
        int c = (linear_idx / output_length) % channels;
        int o = linear_idx % output_length;
        
        // Calculate input start position
        int i_start = o * stride - padding;
        
        // Initialize with minimum float value
        float max_val = -FLT_MAX;
        
        // Perform max pooling
        for (int k = 0; k < kernel_size; k++) {
            int i_pos = i_start + k * dilation;
            
            // Check if the position is valid (not padding)
            if (i_pos >= 0 && i_pos < input_length) {
                int input_idx = b * channels * input_length + c * input_length + i_pos;
                float val = input[input_idx];
                max_val = fmaxf(max_val, val);
            }
        }
        
        // Write output
        output[linear_idx] = max_val;
    }
}

// Optimized kernel for benchmark case (kernel_size=4, stride=2, padding=2, dilation=3)
__global__ void max_pool1d_benchmark_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int channels,
    const int input_length,
    const int output_length) {
    
    // Calculate indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // Hardcoded parameters for benchmark case
    const int kernel_size = 4;
    const int stride = 2;
    const int padding = 2;
    const int dilation = 3;
    
    // Each thread processes multiple elements
    for (int linear_idx = idx; linear_idx < batch_size * channels * output_length; linear_idx += total_threads) {
        // Convert linear index to 3D coordinates
        int b = linear_idx / (channels * output_length);
        int c = (linear_idx / output_length) % channels;
        int o = linear_idx % output_length;
        
        // Calculate input start position
        int i_start = o * stride - padding;
        
        // Initialize with minimum float value
        float max_val = -FLT_MAX;
        
        // Unrolled loop for kernel_size=4
        // Position 0
        int i_pos0 = i_start + 0 * dilation;
        if (i_pos0 >= 0 && i_pos0 < input_length) {
            int input_idx = b * channels * input_length + c * input_length + i_pos0;
            max_val = fmaxf(max_val, input[input_idx]);
        }
        
        // Position 1
        int i_pos1 = i_start + 1 * dilation;
        if (i_pos1 >= 0 && i_pos1 < input_length) {
            int input_idx = b * channels * input_length + c * input_length + i_pos1;
            max_val = fmaxf(max_val, input[input_idx]);
        }
        
        // Position 2
        int i_pos2 = i_start + 2 * dilation;
        if (i_pos2 >= 0 && i_pos2 < input_length) {
            int input_idx = b * channels * input_length + c * input_length + i_pos2;
            max_val = fmaxf(max_val, input[input_idx]);
        }
        
        // Position 3
        int i_pos3 = i_start + 3 * dilation;
        if (i_pos3 >= 0 && i_pos3 < input_length) {
            int input_idx = b * channels * input_length + c * input_length + i_pos3;
            max_val = fmaxf(max_val, input[input_idx]);
        }
        
        // Write output
        output[linear_idx] = max_val;
    }
}
'''

# Try to load CUDA kernel
try:
    from torch.utils.cpp_extension import load_inline
    max_pool1d_cuda = load_inline(
        name='max_pool1d_cuda',
        cpp_sources='',
        cuda_sources=kernel_code,
        functions=['max_pool1d_kernel', 'max_pool1d_benchmark_kernel'],
        with_cuda=True,
        verbose=False
    )
    CUDA_AVAILABLE = True
except Exception as e:
    print(f"CUDA kernel compilation failed: {e}")
    CUDA_AVAILABLE = False

class MaxPool1dFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, dilation):
        # Save for backward
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.input_shape = input.shape
        
        # Get dimensions
        batch_size, channels, input_length = input.shape
        
        # Calculate output length
        output_length = math.floor((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        
        # Allocate output tensor
        output = torch.empty((batch_size, channels, output_length), dtype=input.dtype, device=input.device)
        
        # Use custom CUDA kernel if available and input is on CUDA
        if CUDA_AVAILABLE and input.is_cuda:
            # Determine optimal block and grid dimensions
            threads_per_block = 256
            blocks = min(1024, (batch_size * channels * output_length + threads_per_block - 1) // threads_per_block)
            
            # Check if we're using the benchmark parameters
            if kernel_size == 4 and stride == 2 and padding == 2 and dilation == 3:
                # Use specialized benchmark kernel
                max_pool1d_cuda.max_pool1d_benchmark_kernel(
                    input.contiguous(), output,
                    batch_size, channels, input_length, output_length,
                    grid=(blocks, 1, 1), block=(threads_per_block, 1, 1)
                )
            else:
                # Use general kernel
                max_pool1d_cuda.max_pool1d_kernel(
                    input.contiguous(), output,
                    batch_size, channels, input_length, output_length,
                    kernel_size, stride, padding, dilation,
                    grid=(blocks, 1, 1), block=(threads_per_block, 1, 1)
                )
            
            # Save indices for backward pass (we don't actually compute them)
            ctx.save_for_backward(input)
            
            return output
        else:
            # Fallback to PyTorch implementation
            return F.max_pool1d(input, kernel_size, stride, padding, dilation)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Fallback to PyTorch's implementation for backward pass
        # This is a simplified implementation that doesn't fully match PyTorch's behavior
        # In a real implementation, we would need to track the max indices
        input, = ctx.saved_tensors
        
        # Use PyTorch's implementation for backward
        with torch.enable_grad():
            input_requires_grad = input.requires_grad
            input = input.detach().requires_grad_(True)
            output = F.max_pool1d(input, ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation)
            grad_input = torch.autograd.grad(output, input, grad_output)[0]
        
        return grad_input, None, None, None, None

class ModelNew(nn.Module):
    """
    Optimized implementation of Max Pooling 1D using a custom CUDA kernel.
    
    Args:
        kernel_size (int): Size of the window to take a max over.
        stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
        padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        
        # Cache parameters
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        
        # If return_indices is True, we need to use PyTorch's implementation
        if return_indices:
            self.maxpool = nn.MaxPool1d(
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                dilation=dilation,
                return_indices=True
            )
            self.forward = self._forward_with_indices
        else:
            # For non-indices case, select between custom CUDA kernel and PyTorch's implementation
            if CUDA_AVAILABLE:
                self.forward = self._forward_custom_cuda
            else:
                # Fallback to optimized PyTorch implementation
                if kernel_size == 4 and self.stride == 2 and padding == 2 and dilation == 3:
                    # Benchmark case with hardcoded parameters
                    self.forward = lambda x: F.max_pool1d(x, 4, 2, 2, 3)
                else:
                    # General case
                    self.forward = self._forward_pytorch
    
    def _forward_with_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for indices case."""
        return self.maxpool(x)
    
    def _forward_custom_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using custom CUDA kernel."""
        return MaxPool1dFunction.apply(x, self.kernel_size, self.stride, self.padding, self.dilation)
    
    def _forward_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using PyTorch's implementation."""
        return F.max_pool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)
    
    # This forward method will be replaced at initialization time
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This is a placeholder that will be replaced by one of the specialized
        implementations during initialization.
        """
        pass

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
features = 64
sequence_length = 128
kernel_size = 4
stride = 2
padding = 2
dilation = 3
return_indices = False

def get_inputs():
    x = torch.randn(batch_size, features, sequence_length)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]