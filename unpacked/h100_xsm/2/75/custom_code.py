import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load
import os
import tempfile

# Create a temporary directory for the CUDA extension
temp_dir = tempfile.mkdtemp()

# Write the CUDA kernel code to a file
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void gemm_groupnorm_min_bias_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    const scalar_t* __restrict__ final_bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features,
    int num_groups,
    float eps) {
    
    // Each thread block handles one batch element and one group
    int batch_idx = blockIdx.x;
    int group_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    int features_per_group = out_features / num_groups;
    int group_start = group_idx * features_per_group;
    
    // Shared memory for partial sums and intermediate results
    extern __shared__ float shared_mem[];
    float* group_sum = shared_mem;
    float* group_sum_sq = &shared_mem[blockDim.x];
    float* group_output = &shared_mem[2 * blockDim.x];
    
    // Initialize shared memory
    group_sum[tid] = 0.0f;
    group_sum_sq[tid] = 0.0f;
    
    // Step 1: GEMM operation for this batch and group
    for (int feature_offset = tid; feature_offset < features_per_group; feature_offset += blockDim.x) {
        int feature_idx = group_start + feature_offset;
        float result = bias[feature_idx];
        
        for (int i = 0; i < in_features; ++i) {
            result += input[batch_idx * in_features + i] * weight[feature_idx * in_features + i];
        }
        
        // Store result in shared memory for group norm calculation
        group_output[feature_offset] = result;
        group_sum[tid] += result;
        group_sum_sq[tid] += result * result;
    }
    
    // Parallel reduction for sum and sum of squares
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            group_sum[tid] += group_sum[tid + stride];
            group_sum_sq[tid] += group_sum_sq[tid + stride];
        }
    }
    
    __syncthreads();
    
    // Step 2: Group Normalization
    if (tid == 0) {
        float mean = group_sum[0] / features_per_group;
        float var = (group_sum_sq[0] / features_per_group) - (mean * mean);
        var = max(var, 0.0f);  // Ensure variance is non-negative
        float inv_std = 1.0f / sqrt(var + eps);
        
        // Normalize and apply gamma/beta for each feature in the group
        for (int feature_offset = 0; feature_offset < features_per_group; ++feature_offset) {
            int feature_idx = group_start + feature_offset;
            float normalized = (group_output[feature_offset] - mean) * inv_std;
            group_output[feature_offset] = normalized * gamma[feature_idx] + beta[feature_idx];
        }
    }
    
    __syncthreads();
    
    // Step 3: Find minimum value for this batch across all features in this group
    float local_min = INFINITY;
    for (int feature_offset = tid; feature_offset < features_per_group; feature_offset += blockDim.x) {
        local_min = min(local_min, group_output[feature_offset]);
    }
    
    // Reduce to find minimum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            local_min = min(local_min, shared_mem[tid + stride]);
        }
        shared_mem[tid] = local_min;
    }
    
    __syncthreads();
    
    // Step 4: Add bias and write output
    if (tid == 0) {
        float result = shared_mem[0] + final_bias[group_idx];
        output[batch_idx * num_groups + group_idx] = result;
    }
}

std::vector<torch::Tensor> gemm_groupnorm_min_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor final_bias,
    int num_groups,
    float eps) {
    
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, num_groups}, input.options());
    
    const int threads = 256;
    const int shared_mem_size = 3 * threads * sizeof(float);
    
    const dim3 blocks(batch_size, num_groups);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "gemm_groupnorm_min_bias_cuda", ([&] {
        gemm_groupnorm_min_bias_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            final_bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_features,
            out_features,
            num_groups,
            eps);
    }));
    
    return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gemm_groupnorm_min_bias_cuda, "GEMM GroupNorm Min Bias forward (CUDA)");
}
"""

with open(os.path.join(temp_dir, "gemm_groupnorm_min_bias_cuda.cpp"), "w") as f:
    f.write(cuda_source)

# Try to load the custom CUDA kernel
try:
    fused_ops = load(
        name="gemm_groupnorm_min_bias_cuda",
        sources=[os.path.join(temp_dir, "gemm_groupnorm_min_bias_cuda.cpp")],
        verbose=True,
        build_directory=temp_dir,
        with_cuda=True
    )
    CUSTOM_KERNEL_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load custom CUDA kernel: {e}")
    CUSTOM_KERNEL_AVAILABLE = False

class ModelNew(nn.Module):
    """
    Optimized implementation of the model that performs GEMM, Group Normalization,
    Minimum operation, and Bias addition.
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        num_groups (int): Number of groups for GroupNorm
        bias_shape (tuple): Shape of the bias tensor
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store dimensions for reshaping operations
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.features_per_group = out_features // num_groups
        
        # Ensure all parameters are contiguous for optimal memory access
        self.gemm.weight.data = self.gemm.weight.data.contiguous()
        if self.gemm.bias is not None:
            self.gemm.bias.data = self.gemm.bias.data.contiguous()
        self.group_norm.weight.data = self.group_norm.weight.data.contiguous()
        self.group_norm.bias.data = self.group_norm.bias.data.contiguous()
        self.bias.data = self.bias.data.contiguous()
        
        # Flag to determine if custom kernel is available and should be used
        self.use_custom_kernel = CUSTOM_KERNEL_AVAILABLE and torch.cuda.is_available()
        
        # CUDA graph related attributes
        self.static_input = None
        self.graph = None
        self.static_output = None
        self.warmup_done = False
        self.last_input_shape = None
        
        # Compile the PyTorch fallback function if torch.compile is available
        if hasattr(torch, 'compile'):
            self.optimized_forward = torch.compile(self._forward_pytorch, fullgraph=True, backend="inductor")
        else:
            self.optimized_forward = self._forward_pytorch
    
    def _forward_custom_kernel(self, x):
        """Forward pass using custom CUDA kernel"""
        # Reshape bias to match the expected format for the kernel
        reshaped_bias = self.bias.view(self.num_groups)
        
        # Call our custom fused kernel
        output = fused_ops.forward(
            x,
            self.gemm.weight,
            self.gemm.bias,
            self.group_norm.weight,
            self.group_norm.bias,
            reshaped_bias,
            self.num_groups,
            self.group_norm.eps
        )[0]
        
        # Reshape output to match the expected format (batch_size, out_features, 1, 1)
        return output.view(x.shape[0], 1, 1, self.num_groups)
    
    def _forward_pytorch(self, x):
        """Forward pass using PyTorch operations"""
        # Step 1: GEMM operation
        x = self.gemm(x)
        
        # Step 2: Group Normalization
        # Reshape for group norm if input is 2D
        if x.dim() == 2:
            batch_size, features = x.shape
            x = x.view(batch_size, features, 1, 1)
            x = self.group_norm(x)
            x = x.view(batch_size, features)
        else:
            x = self.group_norm(x)
        
        # Step 3: Min operation - use torch.amin for better performance
        x = torch.amin(x, dim=1, keepdim=True)
        
        # Step 4: Bias addition
        x = x + self.bias
        
        return x
    
    def forward(self, x):
        # Ensure input is contiguous for optimal memory access
        x = x.contiguous()
        
        # Use custom kernel if available and input is on CUDA
        if self.use_custom_kernel and x.is_cuda:
            try:
                return self._forward_custom_kernel(x)
            except Exception as e:
                print(f"Custom kernel failed, falling back to PyTorch: {e}")
                self.use_custom_kernel = False
        
        # Use CUDA graphs for repeated forward passes with same input shape
        if torch.cuda.is_available() and x.is_cuda:
            current_shape = x.shape
            
            # If input shape changed or first run, reset graph
            if self.last_input_shape != current_shape:
                self.static_input = None
                self.graph = None
                self.static_output = None
                self.warmup_done = False
                self.last_input_shape = current_shape
            
            try:
                if not self.warmup_done:
                    # Do more warmup iterations to stabilize performance
                    for _ in range(5):
                        self.optimized_forward(x)
                    self.warmup_done = True
                    
                    # Initialize CUDA graph
                    self.static_input = torch.zeros_like(x, device=x.device)
                    self.graph = torch.cuda.CUDAGraph()
                    
                    # Capture the graph
                    with torch.cuda.graph(self.graph):
                        self.static_input.copy_(x)
                        self.static_output = self.optimized_forward(self.static_input)
                
                # Run the captured graph with new input data
                self.static_input.copy_(x)
                self.graph.replay()
                return self.static_output
                
            except Exception:
                # If CUDA graph fails for any reason, fall back to regular execution
                pass
        
        # Fallback for CPU or when CUDA is not available or CUDA graph failed
        return self.optimized_forward(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 512
out_features = 256
num_groups = 8
bias_shape = (1, out_features, 1, 1)

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_features, out_features, num_groups, bias_shape]