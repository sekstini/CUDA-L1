import torch
import torch.nn as nn
import math

# Optimized CUDA kernel for GEMM with fused operations
cuda_kernel_code = """
extern "C" __global__ void gemm_subtract_avgpool_gelu_residual(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias_minus_subtract,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features)
{
    // Block dimensions
    constexpr int BM = 32;  // Threads per block in M dimension
    constexpr int BN = 8;   // Threads per block in N dimension
    constexpr int BK = 32;  // Block tile size in K dimension
    
    // Thread tiling factors (each thread computes a 4x2 output tile)
    constexpr int TM = 4;
    constexpr int TN = 2;
    
    // Shared memory tiles (with padding to avoid bank conflicts)
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN * TN + 1];
    
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * BN + tx;
    
    // Starting indices for this block
    int row_start = by * BM;
    int col_start = bx * BN * TN;
    
    // Registers for accumulating results
    float accum[TM][TN] = {0.0f};
    
    // Registers for storing input elements
    float a_reg[TM];
    float b_reg[TN];
    
    // Loop over tiles in K dimension
    for (int k_tile = 0; k_tile < (in_features + BK - 1) / BK; ++k_tile) {
        // Load input tile into shared memory
        for (int tm = 0; tm < TM; tm++) {
            int row = row_start + tid / (BK/TM) * TM + tm;
            int col = k_tile * BK + tid % (BK/TM);
            
            if (row < batch_size && col < in_features) {
                As[tid / (BK/TM) * TM + tm][tid % (BK/TM)] = input[row * in_features + col];
            } else {
                As[tid / (BK/TM) * TM + tm][tid % (BK/TM)] = 0.0f;
            }
        }
        
        // Load weight tile into shared memory
        for (int tn = 0; tn < TN; tn++) {
            int row = k_tile * BK + tid / (BN*TN);
            int col = col_start + tid % (BN*TN) + tn * BN;
            
            if (row < in_features && col < out_features) {
                Bs[tid / (BN*TN)][tid % (BN*TN) + tn * BN] = weight[row * out_features + col];
            } else {
                Bs[tid / (BN*TN)][tid % (BN*TN) + tn * BN] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial products for this tile
        for (int k = 0; k < BK; k++) {
            // Load data from shared memory to registers
            for (int tm = 0; tm < TM; tm++) {
                a_reg[tm] = As[ty * TM + tm][k];
            }
            
            for (int tn = 0; tn < TN; tn++) {
                b_reg[tn] = Bs[k][tx + tn * BN];
            }
            
            // Compute outer product
            for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) {
                    accum[tm][tn] += a_reg[tm] * b_reg[tn];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Add bias and perform subtract
    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
            int col = col_start + tx + tn * BN;
            if (col < out_features) {
                accum[tm][tn] += bias_minus_subtract[col];
            }
        }
    }
    
    // Perform global average pooling using warp-level reduction
    float row_sums[TM] = {0.0f};
    
    for (int tm = 0; tm < TM; tm++) {
        int row = row_start + ty * TM + tm;
        if (row < batch_size) {
            for (int tn = 0; tn < TN; tn++) {
                int col = col_start + tx + tn * BN;
                if (col < out_features) {
                    row_sums[tm] += accum[tm][tn];
                }
            }
        }
    }
    
    // Warp-level reduction for each row
    for (int offset = 16; offset > 0; offset /= 2) {
        for (int tm = 0; tm < TM; tm++) {
            row_sums[tm] += __shfl_down_sync(0xffffffff, row_sums[tm], offset);
        }
    }
    
    // Block-level reduction using shared memory
    __shared__ float block_sums[BM];
    
    for (int tm = 0; tm < TM; tm++) {
        int row = row_start + ty * TM + tm;
        if (row < batch_size && tx == 0) {
            block_sums[ty * TM + tm] = row_sums[tm];
        }
    }
    
    __syncthreads();
    
    // Final reduction and division
    if (tid < batch_size && tid < BM && bx == 0) {
        int row = row_start + tid;
        if (row < batch_size) {
            float avg = block_sums[tid] / out_features;
            
            // GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            float x = avg;
            float x3 = x * x * x;
            float inner = 0.7978845608028654f * (x + 0.044715f * x3);
            float gelu_val = x * 0.5f * (1.0f + tanhf(inner));
            
            // Write result to output (first element of each row) and copy original input
            output[row * in_features] = input[row * in_features] + gelu_val;
            
            // Copy rest of original input
            for (int i = 1; i < in_features; i++) {
                output[row * in_features + i] = input[row * in_features + i];
            }
        }
    }
}
"""

class ModelNew(nn.Module):
    """
    Optimized model that performs a series of operations: Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias (bool): Whether to use bias in the linear layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.in_features = in_features
        self.out_features = out_features
        
        # Pre-compute transposed weight for faster matrix multiplication in PyTorch fallback
        self.register_buffer('weight_t', self.gemm.weight.t().contiguous())
        
        # Pre-compute bias minus subtract for efficiency
        if bias and self.gemm.bias is not None:
            self.register_buffer('bias_minus_subtract', self.gemm.bias - self.subtract)
        else:
            self.register_buffer('bias_minus_subtract', -self.subtract)
        
        # Pre-allocate buffers for fallback path
        self.register_buffer('gemm_output', torch.zeros(batch_size, out_features))
        self.register_buffer('mean_output', torch.zeros(batch_size, 1))
        
        # Compile CUDA kernel
        self.cuda_kernel = None
        if torch.cuda.is_available():
            try:
                from torch.utils.cpp_extension import load_inline
                self.cuda_kernel = load_inline(
                    name="gemm_fused_ops",
                    cpp_sources="",
                    cuda_sources=cuda_kernel_code,
                    functions=["gemm_subtract_avgpool_gelu_residual"],
                    with_cuda=True,
                    verbose=False,
                    extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++14']
                )
            except Exception as e:
                print(f"Warning: Failed to compile CUDA kernel: {e}")
                self.cuda_kernel = None
        
        # Register parameter update hooks
        def update_weight_t(grad):
            if self.training:
                with torch.no_grad():
                    self.weight_t.copy_(self.gemm.weight.t().contiguous())
            return grad
        
        def update_bias_subtract(grad):
            if self.training:
                with torch.no_grad():
                    if hasattr(self.gemm, 'bias') and self.gemm.bias is not None:
                        self.bias_minus_subtract.copy_(self.gemm.bias - self.subtract)
                    else:
                        self.bias_minus_subtract.copy_(-self.subtract)
            return grad
        
        self.gemm.weight.register_hook(update_weight_t)
        if bias and self.gemm.bias is not None:
            self.gemm.bias.register_hook(update_bias_subtract)
        self.subtract.register_hook(update_bias_subtract)
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor after all operations
        """
        # Store reference to original input (no clone needed)
        original_x = x
        
        # Ensure input is contiguous
        if not x.is_contiguous():
            x = x.contiguous()
        
        batch_size_actual = x.size(0)
        
        # Use custom CUDA kernel for inference on GPU with correct batch size
        if (self.cuda_kernel is not None and 
            x.is_cuda and 
            batch_size_actual == batch_size and
            not self.training):
            
            output = torch.empty_like(original_x)
            
            # Calculate grid and block dimensions
            threads_per_block = (8, 8)  # 8x8 = 64 threads per block
            blocks_per_grid = (
                (self.out_features + 15) // 16,  # Ceiling division for output features
                (batch_size_actual + 31) // 32    # Ceiling division for batch size
            )
            
            # Launch kernel
            self.cuda_kernel.gemm_subtract_avgpool_gelu_residual(
                x,                      # input
                self.gemm.weight,       # weight
                self.bias_minus_subtract,  # bias_minus_subtract
                output,                 # output
                batch_size_actual,      # batch_size
                self.in_features,       # in_features
                self.out_features       # out_features
            )
            
            return output
        
        # Fallback to optimized PyTorch implementation
        if batch_size_actual == batch_size and x.device == self.gemm_output.device:
            # Optimized GEMM operation
            torch.addmm(self.bias_minus_subtract, x, self.weight_t, out=self.gemm_output)
            
            # GlobalAvgPool
            torch.mean(self.gemm_output, dim=1, keepdim=True, out=self.mean_output)
            
            # LogSumExp (for a single value per batch item, logsumexp is just the value itself)
            # GELU
            x = torch.nn.functional.gelu(self.mean_output)
            
            # ResidualAdd
            return x + original_x
        else:
            # General fallback path for different batch sizes or devices
            x = self.gemm(x)
            x = x - self.subtract
            x = torch.mean(x, dim=1, keepdim=True)
            x = torch.logsumexp(x, dim=1, keepdim=True)  # For single values, this is equivalent to the value itself
            x = torch.nn.functional.gelu(x)
            return x + original_x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]