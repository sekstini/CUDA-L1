import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define CUDA kernels for fused operations
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fused BatchNorm + ReLU kernel with optimized memory access
template <typename scalar_t>
__global__ void fused_batchnorm_relu_kernel(
    const scalar_t* __restrict__ input, 
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ weight, 
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ running_mean, 
    const scalar_t* __restrict__ running_var,
    int batch_size, int channels, int height, int width,
    float eps) {
    
    // Use shared memory for batch norm parameters with padding to avoid bank conflicts
    extern __shared__ char shared_mem[];
    scalar_t* s_weight = (scalar_t*)shared_mem;
    scalar_t* s_bias = s_weight + channels + (channels % 32 == 0 ? 1 : 0);
    scalar_t* s_mean = s_bias + channels + (channels % 32 == 0 ? 1 : 0);
    scalar_t* s_inv_std = s_mean + channels + (channels % 32 == 0 ? 1 : 0);
    
    // Load batch norm parameters into shared memory
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        s_weight[i] = weight[i];
        s_bias[i] = bias[i];
        s_mean[i] = running_mean[i];
        s_inv_std[i] = rsqrtf(running_var[i] + eps);
    }
    __syncthreads();
    
    // Calculate global thread index with grid-stride loop
    const int n_elements = batch_size * channels * height * width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process elements with grid-stride loop for better workload distribution
    for (; idx < n_elements; idx += stride) {
        const int w = idx % width;
        const int h = (idx / width) % height;
        const int c = (idx / (width * height)) % channels;
        const int n = idx / (width * height * channels);
        
        // Cache channel parameters in registers for faster access
        const scalar_t w_c = s_weight[c];
        const scalar_t b_c = s_bias[c];
        const scalar_t mean_c = s_mean[c];
        const scalar_t inv_std_c = s_inv_std[c];
        
        // Compute normalized value
        const scalar_t val = input[idx];
        const scalar_t normalized = (val - mean_c) * inv_std_c;
        const scalar_t scaled = normalized * w_c + b_c;
        
        // Apply ReLU and store result
        output[idx] = scaled > 0 ? scaled : 0;
    }
}

// Fused Add + ReLU kernel with vectorized memory access
template <typename scalar_t>
__global__ void fused_add_relu_kernel(
    const scalar_t* __restrict__ input1, 
    const scalar_t* __restrict__ input2, 
    scalar_t* __restrict__ output,
    int size) {
    
    // Calculate global thread index with grid-stride loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at once when possible for better memory throughput
    const int vec_size = size / 4 * 4;
    
    // Vector processing for aligned data
    for (; idx < vec_size; idx += stride * 4) {
        if (idx + 3 < size) {
            // Check alignment for vectorized load
            if (((uintptr_t)&input1[idx] % 16 == 0) && ((uintptr_t)&input2[idx] % 16 == 0) && 
                ((uintptr_t)&output[idx] % 16 == 0)) {
                
                float4 a = *reinterpret_cast<const float4*>(&input1[idx]);
                float4 b = *reinterpret_cast<const float4*>(&input2[idx]);
                
                float4 result;
                result.x = a.x + b.x;
                result.y = a.y + b.y;
                result.z = a.z + b.z;
                result.w = a.w + b.w;
                
                result.x = result.x > 0 ? result.x : 0;
                result.y = result.y > 0 ? result.y : 0;
                result.z = result.z > 0 ? result.z : 0;
                result.w = result.w > 0 ? result.w : 0;
                
                *reinterpret_cast<float4*>(&output[idx]) = result;
            } else {
                // Fallback for unaligned memory
                for (int i = 0; i < 4 && idx + i < size; ++i) {
                    const scalar_t sum = input1[idx + i] + input2[idx + i];
                    output[idx + i] = sum > 0 ? sum : 0;
                }
            }
        }
    }
    
    // Handle remaining elements
    for (idx = vec_size + threadIdx.x + blockIdx.x * blockDim.x; idx < size; idx += stride) {
        const scalar_t sum = input1[idx] + input2[idx];
        output[idx] = sum > 0 ? sum : 0;
    }
}

// C++ wrapper for the BatchNorm + ReLU kernel
torch::Tensor fused_batchnorm_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps) {
    
    // Ensure input is contiguous
    input = input.contiguous();
    
    // Create output tensor
    auto output = torch::empty_like(input);
    
    // Get dimensions
    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    // Calculate kernel launch parameters
    int threads_per_block = 256;
    int total_elements = batch_size * channels * height * width;
    int blocks = std::min(1024, (total_elements + threads_per_block - 1) / threads_per_block);
    
    // Calculate shared memory size with padding to avoid bank conflicts
    int shared_mem_size = channels * 4 * sizeof(float);  // 4 arrays of size channels
    if (channels % 32 == 0) {
        shared_mem_size += 3 * sizeof(float); // Add padding
    }
    
    // Launch kernel with dynamic shared memory
    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_batchnorm_relu_cuda", ([&] {
        fused_batchnorm_relu_kernel<scalar_t><<<blocks, threads_per_block, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            running_mean.data_ptr<scalar_t>(),
            running_var.data_ptr<scalar_t>(),
            batch_size, channels, height, width, eps
        );
    }));
    
    return output;
}

// C++ wrapper for the Add + ReLU kernel
torch::Tensor fused_add_relu_cuda(
    torch::Tensor input1,
    torch::Tensor input2) {
    
    // Ensure inputs are contiguous
    input1 = input1.contiguous();
    input2 = input2.contiguous();
    
    // Create output tensor
    auto output = torch::empty_like(input1);
    
    // Get size
    int size = input1.numel();
    
    // Calculate kernel launch parameters
    int threads_per_block = 256;
    int blocks = std::min(1024, (size + threads_per_block - 1) / threads_per_block);
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input1.type(), "fused_add_relu_cuda", ([&] {
        fused_add_relu_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input1.data_ptr<scalar_t>(),
            input2.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));
    
    return output;
}

// Optimized ResNet basic block implementation
std::vector<torch::Tensor> resnet_basic_block_forward_cuda(
    torch::Tensor x,
    torch::Tensor conv1_weight,
    torch::Tensor bn1_weight,
    torch::Tensor bn1_bias,
    torch::Tensor bn1_running_mean,
    torch::Tensor bn1_running_var,
    torch::Tensor conv2_weight,
    torch::Tensor bn2_weight,
    torch::Tensor bn2_bias,
    torch::Tensor bn2_running_mean,
    torch::Tensor bn2_running_var,
    torch::Tensor downsample_conv_weight,
    torch::Tensor downsample_bn_weight,
    torch::Tensor downsample_bn_bias,
    torch::Tensor downsample_bn_running_mean,
    torch::Tensor downsample_bn_running_var,
    int stride,
    float eps) {
    
    // Ensure input is contiguous
    x = x.contiguous();
    
    // Create CUDA streams for parallel execution of main path and downsample path
    at::cuda::CUDAStreamGuard guard1(at::cuda::getStreamFromPool());
    auto stream_main = at::cuda::getCurrentCUDAStream();
    
    at::cuda::CUDAStreamGuard guard2(at::cuda::getStreamFromPool());
    auto stream_downsample = at::cuda::getCurrentCUDAStream();
    
    // Main path
    at::cuda::setCurrentCUDAStream(stream_main);
    auto out = torch::conv2d(x, conv1_weight, {}, stride, 1);
    out = fused_batchnorm_relu_cuda(
        out, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var, eps);
    out = torch::conv2d(out, conv2_weight, {}, 1, 1);
    out = torch::batch_norm(
        out, bn2_running_mean, bn2_running_var, bn2_weight, bn2_bias, 
        false, 0.1, eps);
    
    // Downsample path in parallel
    at::cuda::setCurrentCUDAStream(stream_downsample);
    auto identity = torch::conv2d(x, downsample_conv_weight, {}, stride, 0);
    identity = torch::batch_norm(
        identity, downsample_bn_running_mean, downsample_bn_running_var,
        downsample_bn_weight, downsample_bn_bias, false, 0.1, eps);
    
    // Synchronize streams before add
    at::cuda::setCurrentCUDAStream(stream_main);
    cudaStreamSynchronize(stream_downsample.stream());
    
    // Add identity and apply ReLU (fused)
    out = fused_add_relu_cuda(out, identity);
    
    return {out};
}
"""

cpp_source = """
#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions
torch::Tensor fused_batchnorm_relu_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps);

torch::Tensor fused_add_relu_cuda(
    torch::Tensor input1,
    torch::Tensor input2);

std::vector<torch::Tensor> resnet_basic_block_forward_cuda(
    torch::Tensor x,
    torch::Tensor conv1_weight,
    torch::Tensor bn1_weight,
    torch::Tensor bn1_bias,
    torch::Tensor bn1_running_mean,
    torch::Tensor bn1_running_var,
    torch::Tensor conv2_weight,
    torch::Tensor bn2_weight,
    torch::Tensor bn2_bias,
    torch::Tensor bn2_running_mean,
    torch::Tensor bn2_running_var,
    torch::Tensor downsample_conv_weight,
    torch::Tensor downsample_bn_weight,
    torch::Tensor downsample_bn_bias,
    torch::Tensor downsample_bn_running_mean,
    torch::Tensor downsample_bn_running_var,
    int stride,
    float eps);

// Wrapper functions with input validation
torch::Tensor fused_batchnorm_relu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps) {
    
    // Check that all inputs are on CUDA
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.device().is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(running_mean.device().is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.device().is_cuda(), "running_var must be a CUDA tensor");
    
    return fused_batchnorm_relu_cuda(input, weight, bias, running_mean, running_var, eps);
}

torch::Tensor fused_add_relu(
    torch::Tensor input1,
    torch::Tensor input2) {
    
    // Check that inputs are on CUDA
    TORCH_CHECK(input1.device().is_cuda(), "input1 must be a CUDA tensor");
    TORCH_CHECK(input2.device().is_cuda(), "input2 must be a CUDA tensor");
    TORCH_CHECK(input1.sizes() == input2.sizes(), "input shapes must match");
    
    return fused_add_relu_cuda(input1, input2);
}

std::vector<torch::Tensor> resnet_basic_block_forward(
    torch::Tensor x,
    torch::Tensor conv1_weight,
    torch::Tensor bn1_weight,
    torch::Tensor bn1_bias,
    torch::Tensor bn1_running_mean,
    torch::Tensor bn1_running_var,
    torch::Tensor conv2_weight,
    torch::Tensor bn2_weight,
    torch::Tensor bn2_bias,
    torch::Tensor bn2_running_mean,
    torch::Tensor bn2_running_var,
    torch::Tensor downsample_conv_weight,
    torch::Tensor downsample_bn_weight,
    torch::Tensor downsample_bn_bias,
    torch::Tensor downsample_bn_running_mean,
    torch::Tensor downsample_bn_running_var,
    int stride,
    float eps) {
    
    // Check that all inputs are on CUDA
    TORCH_CHECK(x.device().is_cuda(), "input must be a CUDA tensor");
    
    return resnet_basic_block_forward_cuda(
        x, conv1_weight, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
        conv2_weight, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var,
        downsample_conv_weight, downsample_bn_weight, downsample_bn_bias,
        downsample_bn_running_mean, downsample_bn_running_var, stride, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_batchnorm_relu", &fused_batchnorm_relu, "Fused BatchNorm + ReLU");
    m.def("fused_add_relu", &fused_add_relu, "Fused Add + ReLU");
    m.def("resnet_basic_block_forward", &resnet_basic_block_forward, "Optimized ResNet Basic Block Forward");
}
"""

# Create a singleton class to load CUDA extension only once
class CUDAExtensionLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CUDAExtensionLoader, cls).__new__(cls)
            cls._instance.resnet_cuda = None
            cls._instance.load_extension()
        return cls._instance
    
    def load_extension(self):
        try:
            self.resnet_cuda = load_inline(
                name='resnet_cuda',
                cpp_sources=[cpp_source],
                cuda_sources=[cuda_source],
                functions=['fused_batchnorm_relu', 'fused_add_relu', 'resnet_basic_block_forward'],
                verbose=False,
                with_cuda=True
            )
        except Exception as e:
            print(f"Could not load CUDA extension: {e}")
            self.resnet_cuda = None
    
    def get_extension(self):
        return self.resnet_cuda

class ResNetBasicBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, conv1_weight, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
                conv2_weight, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var,
                downsample_conv_weight, downsample_bn_weight, downsample_bn_bias,
                downsample_bn_running_mean, downsample_bn_running_var, stride, eps):
        
        resnet_cuda = CUDAExtensionLoader().get_extension()
        
        if resnet_cuda is not None:
            return resnet_cuda.resnet_basic_block_forward(
                x, conv1_weight, bn1_weight, bn1_bias, bn1_running_mean, bn1_running_var,
                conv2_weight, bn2_weight, bn2_bias, bn2_running_mean, bn2_running_var,
                downsample_conv_weight, downsample_bn_weight, downsample_bn_bias,
                downsample_bn_running_mean, downsample_bn_running_var, stride, eps)[0]
        else:
            # Fallback implementation using PyTorch ops
            identity = x
            
            # First conv + bn + relu
            out = F.conv2d(x, conv1_weight, bias=None, stride=stride, padding=1)
            out = F.batch_norm(out, bn1_running_mean, bn1_running_var, bn1_weight, bn1_bias, False, 0.1, eps)
            out = F.relu(out, inplace=True)
            
            # Second conv + bn
            out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
            out = F.batch_norm(out, bn2_running_mean, bn2_running_var, bn2_weight, bn2_bias, False, 0.1, eps)
            
            # Downsample
            identity = F.conv2d(x, downsample_conv_weight, bias=None, stride=stride)
            identity = F.batch_norm(identity, downsample_bn_running_mean, downsample_bn_running_var,
                                  downsample_bn_weight, downsample_bn_bias, False, 0.1, eps)
            
            # Add + relu
            out = out + identity
            out = F.relu(out, inplace=True)
            
            return out

class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        """
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride
        
        # Pre-allocate CUDA streams for potential overlap
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # Load CUDA extension singleton
        if torch.cuda.is_available():
            CUDAExtensionLoader()

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        # Use optimized implementation if on CUDA
        if x.is_cuda:
            # Ensure input is contiguous for best performance
            if not x.is_contiguous():
                x = x.contiguous()
                
            # Use our optimized CUDA implementation with stream for potential overlap
            with torch.cuda.stream(self.stream):
                return ResNetBasicBlockFunction.forward(
                    None,  # ctx is not used in forward
                    x, 
                    self.conv1.weight, self.bn1.weight, self.bn1.bias, 
                    self.bn1.running_mean, self.bn1.running_var,
                    self.conv2.weight, self.bn2.weight, self.bn2.bias, 
                    self.bn2.running_mean, self.bn2.running_var,
                    self.downsample[0].weight, self.downsample[1].weight, self.downsample[1].bias,
                    self.downsample[1].running_mean, self.downsample[1].running_var,
                    self.stride, self.bn1.eps
                )
        else:
            # Standard implementation for CPU
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
in_channels = 3
out_channels = 64
stride = 1
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, in_channels, 224, 224)]

def get_init_inputs():
    return [in_channels, out_channels, stride]