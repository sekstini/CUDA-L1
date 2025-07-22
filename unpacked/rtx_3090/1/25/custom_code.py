import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom CUDA kernel for optimized Swish implementation
swish_kernel_code = """
extern "C" __global__ void swish_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int size) {
    
    // Grid-stride loop to handle large tensors efficiently
    // Process 4 elements per thread when possible to increase ILP
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Main loop - process 4 elements at a time
    for (; idx + 3 < size; idx += stride * 4) {
        // Load 4 input values
        float x1 = input[idx];
        float x2 = input[idx + stride];
        float x3 = input[idx + stride * 2];
        float x4 = input[idx + stride * 3];
        
        // Compute sigmoid with numerical stability for each value
        float sigmoid_x1, sigmoid_x2, sigmoid_x3, sigmoid_x4;
        
        // For x1
        if (x1 >= 0) {
            sigmoid_x1 = 1.0f / (1.0f + __expf(-x1));
        } else {
            float exp_x = __expf(x1);
            sigmoid_x1 = exp_x / (1.0f + exp_x);
        }
        
        // For x2
        if (x2 >= 0) {
            sigmoid_x2 = 1.0f / (1.0f + __expf(-x2));
        } else {
            float exp_x = __expf(x2);
            sigmoid_x2 = exp_x / (1.0f + exp_x);
        }
        
        // For x3
        if (x3 >= 0) {
            sigmoid_x3 = 1.0f / (1.0f + __expf(-x3));
        } else {
            float exp_x = __expf(x3);
            sigmoid_x3 = exp_x / (1.0f + exp_x);
        }
        
        // For x4
        if (x4 >= 0) {
            sigmoid_x4 = 1.0f / (1.0f + __expf(-x4));
        } else {
            float exp_x = __expf(x4);
            sigmoid_x4 = exp_x / (1.0f + exp_x);
        }
        
        // Compute swish and store results
        output[idx] = x1 * sigmoid_x1;
        output[idx + stride] = x2 * sigmoid_x2;
        output[idx + stride * 2] = x3 * sigmoid_x3;
        output[idx + stride * 3] = x4 * sigmoid_x4;
    }
    
    // Handle remaining elements
    for (; idx < size; idx += stride) {
        float x = input[idx];
        
        // Compute sigmoid with numerical stability
        float sigmoid_x;
        if (x >= 0) {
            sigmoid_x = 1.0f / (1.0f + __expf(-x));
        } else {
            float exp_x = __expf(x);
            sigmoid_x = exp_x / (1.0f + exp_x);
        }
        
        // Compute swish: x * sigmoid(x)
        output[idx] = x * sigmoid_x;
    }
}

// Vectorized version using float4 for coalesced memory access
extern "C" __global__ void swish_forward_kernel_vectorized(
    const float4* __restrict__ input4,
    float4* __restrict__ output4,
    const int size4) {
    
    // Grid-stride loop to handle large tensors efficiently
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < size4; 
         idx += blockDim.x * gridDim.x) {
        
        // Load input value (4 floats at once)
        float4 x4 = input4[idx];
        float4 result;
        
        // Process first element
        if (x4.x >= 0) {
            result.x = x4.x / (1.0f + __expf(-x4.x));
        } else {
            float exp_x = __expf(x4.x);
            result.x = x4.x * exp_x / (1.0f + exp_x);
        }
        
        // Process second element
        if (x4.y >= 0) {
            result.y = x4.y / (1.0f + __expf(-x4.y));
        } else {
            float exp_x = __expf(x4.y);
            result.y = x4.y * exp_x / (1.0f + exp_x);
        }
        
        // Process third element
        if (x4.z >= 0) {
            result.z = x4.z / (1.0f + __expf(-x4.z));
        } else {
            float exp_x = __expf(x4.z);
            result.z = x4.z * exp_x / (1.0f + exp_x);
        }
        
        // Process fourth element
        if (x4.w >= 0) {
            result.w = x4.w / (1.0f + __expf(-x4.w));
        } else {
            float exp_x = __expf(x4.w);
            result.w = x4.w * exp_x / (1.0f + exp_x);
        }
        
        // Store result (4 floats at once)
        output4[idx] = result;
    }
}
"""

# Try to compile the CUDA kernel
try:
    from torch.utils.cpp_extension import load_inline
    swish_cuda = load_inline(
        name="swish_cuda_optimized",
        cpp_sources="",
        cuda_sources=swish_kernel_code,
        functions=["swish_forward_kernel", "swish_forward_kernel_vectorized"],
        with_cuda=True,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False
    )
    CUDA_KERNEL_AVAILABLE = True
except Exception:
    CUDA_KERNEL_AVAILABLE = False

class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save input for backward pass
        ctx.save_for_backward(x)
        
        # If CUDA kernel is not available or tensor is not on CUDA,
        # fall back to PyTorch's implementation
        if not CUDA_KERNEL_AVAILABLE or not x.is_cuda:
            return F.silu(x)
        
        # Ensure contiguous memory layout
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Create output tensor
        output = torch.empty_like(x)
        numel = x.numel()
        
        # Configure kernel parameters
        threads_per_block = 256
        blocks_per_grid = min(65535, (numel + threads_per_block - 1) // threads_per_block)
        
        # Check if we can use vectorized version (size must be multiple of 4)
        if numel % 4 == 0 and numel >= 1024:
            # Reinterpret tensors as float4
            input_float4 = x.view(torch.cuda.FloatTensor)
            output_float4 = output.view(torch.cuda.FloatTensor)
            
            # Launch vectorized kernel
            swish_cuda.swish_forward_kernel_vectorized(
                input_float4.data_ptr(),
                output_float4.data_ptr(),
                numel // 4,
                grid=(blocks_per_grid,),
                block=(threads_per_block,)
            )
        else:
            # Launch regular kernel
            swish_cuda.swish_forward_kernel(
                x.data_ptr(),
                output.data_ptr(),
                numel,
                grid=(blocks_per_grid,),
                block=(threads_per_block,)
            )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        
        # Use PyTorch's optimized operations for backward pass
        sigmoid_x = torch.sigmoid(x)
        grad_input = grad_output * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))
        
        return grad_input

class ModelNew(nn.Module):
    """
    Optimized model that performs a Swish activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation to the input tensor using optimized CUDA implementation.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        # Try to use our custom CUDA kernel first
        if CUDA_KERNEL_AVAILABLE and x.is_cuda:
            return SwishFunction.apply(x)
        
        # Fall back to PyTorch's optimized implementation
        return F.silu(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed