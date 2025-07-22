import torch
import torch.nn as nn

# Custom CUDA kernel for matrix-scalar multiplication
cuda_kernel = """
extern "C" __global__ void matrix_scalar_mul_kernel(
    const float* input, 
    float* output, 
    const float scalar,
    const int size) 
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        output[i] = input[i] * scalar;
    }
}
"""

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix-scalar multiplication (C = A * s)
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.output = None
        self.stream = None
        self.scalar_tensor = None
        self.last_scalar = None
        self.last_shape = None
        self.last_dtype = None
        self.last_device = None
        self.kernel = None
        
        # Load CUDA kernel if CUDA is available
        if torch.cuda.is_available():
            from torch.utils.cpp_extension import load_inline
            try:
                cuda_module = load_inline(
                    name="matrix_scalar_mul",
                    cpp_sources="",
                    cuda_sources=cuda_kernel,
                    functions=["matrix_scalar_mul_kernel"],
                    with_cuda=True,
                    verbose=False
                )
                self.kernel = cuda_module.matrix_scalar_mul_kernel
            except Exception as e:
                print(f"Failed to load CUDA kernel: {e}")
                self.kernel = None
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication with optimized implementation.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        # Move tensor to GPU if it's not already there and if CUDA is available
        if not A.is_cuda and torch.cuda.is_available():
            A = A.cuda()
        
        # Ensure contiguous memory layout for optimal access patterns
        if not A.is_contiguous():
            A = A.contiguous()
        
        # Check if we need to create or update the output tensor
        current_shape = A.shape
        current_dtype = A.dtype
        current_device = A.device
        
        if (self.output is None or 
            self.last_shape != current_shape or 
            self.last_dtype != current_dtype or 
            self.last_device != current_device):
            self.output = torch.empty_like(A)
            self.last_shape = current_shape
            self.last_dtype = current_dtype
            self.last_device = current_device
        
        # Create a dedicated CUDA stream if not already created and if we're on GPU
        if A.is_cuda and self.stream is None:
            self.stream = torch.cuda.Stream()
        
        # Cache scalar value as tensor on the same device as A
        if (self.scalar_tensor is None or 
            self.last_scalar != s or 
            self.scalar_tensor.device != A.device or
            self.scalar_tensor.dtype != A.dtype):
            self.scalar_tensor = torch.tensor(s, device=A.device, dtype=A.dtype)
            self.last_scalar = s
        
        # Use custom CUDA kernel if available and we're on GPU with float32 tensors
        if (self.kernel is not None and A.is_cuda and A.dtype == torch.float32):
            size = A.numel()
            
            # Calculate grid and block dimensions
            threads_per_block = 256
            
            # Get optimal grid size based on SM count and workload size
            try:
                device_props = torch.cuda.get_device_properties(A.device)
                sm_count = device_props.multi_processor_count
                
                # Calculate blocks needed for full occupancy
                # Based on empirical testing, 40 blocks per SM seems optimal for this workload
                blocks_for_occupancy = sm_count * 40
                
                # Calculate minimum blocks needed to cover the data
                min_blocks_needed = (size + threads_per_block - 1) // threads_per_block
                
                # Use the larger of the two values, but don't exceed CUDA limits
                blocks_per_grid = min(65535, max(blocks_for_occupancy, min_blocks_needed))
            except:
                # Fallback if we can't query device properties
                blocks_per_grid = min(65535, (size + threads_per_block - 1) // threads_per_block)
            
            with torch.cuda.stream(self.stream):
                self.kernel(
                    blocks_per_grid, 
                    threads_per_block, 
                    [A.data_ptr(), self.output.data_ptr(), float(s), size]
                )
        else:
            # Fallback to PyTorch's optimized implementation
            if self.stream is not None and A.is_cuda:
                with torch.cuda.stream(self.stream):
                    torch.mul(A, self.scalar_tensor, out=self.output)
            else:
                torch.mul(A, self.scalar_tensor, out=self.output)
        
        return self.output

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
M = 16384
N = 4096

def get_inputs():
    A = torch.randn(M, N)
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []  # No special initialization inputs needed