import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load
import os

# Define CUDA kernel for FireModule operations
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for fused squeeze operation with ReLU
__global__ void squeeze_relu_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * height * width) return;
    
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % out_channels;
    int b = idx / (width * height * out_channels);
    
    float sum = bias[c];
    for (int ic = 0; ic < in_channels; ++ic) {
        sum += input[b * in_channels * height * width + ic * height * width + h * width + w] * 
               weights[c * in_channels + ic];
    }
    
    // ReLU activation
    output[idx] = sum > 0.0f ? sum : 0.0f;
}

// CUDA kernel for fused expand1x1 operation with ReLU
__global__ void expand1x1_relu_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * height * width) return;
    
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % out_channels;
    int b = idx / (width * height * out_channels);
    
    float sum = bias[c];
    for (int ic = 0; ic < in_channels; ++ic) {
        sum += input[b * in_channels * height * width + ic * height * width + h * width + w] * 
               weights[c * in_channels + ic];
    }
    
    // ReLU activation
    output[idx] = sum > 0.0f ? sum : 0.0f;
}

// CUDA kernel for fused expand3x3 operation with ReLU
__global__ void expand3x3_relu_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * height * width) return;
    
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % out_channels;
    int b = idx / (width * height * out_channels);
    
    float sum = bias[c];
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                int h_in = h - 1 + kh;
                int w_in = w - 1 + kw;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    sum += input[b * in_channels * height * width + ic * height * width + h_in * width + w_in] * 
                           weights[c * in_channels * 9 + ic * 9 + kh * 3 + kw];
                }
            }
        }
    }
    
    // ReLU activation
    output[idx] = sum > 0.0f ? sum : 0.0f;
}

// Function to launch squeeze_relu_kernel
torch::Tensor squeeze_relu_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weights.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, 
                              input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * height * width + threads - 1) / threads;
    
    squeeze_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width);
    
    return output;
}

// Function to launch expand1x1_relu_kernel
torch::Tensor expand1x1_relu_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weights.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, 
                              input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * height * width + threads - 1) / threads;
    
    expand1x1_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width);
    
    return output;
}

// Function to launch expand3x3_relu_kernel
torch::Tensor expand3x3_relu_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weights.size(0);
    
    auto output = torch::zeros({batch_size, out_channels, height, width}, 
                              input.options());
    
    const int threads = 256;
    const int blocks = (batch_size * out_channels * height * width + threads - 1) / threads;
    
    expand3x3_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("squeeze_relu", &squeeze_relu_cuda, "Squeeze operation with ReLU (CUDA)");
    m.def("expand1x1_relu", &expand1x1_relu_cuda, "Expand 1x1 operation with ReLU (CUDA)");
    m.def("expand3x3_relu", &expand3x3_relu_cuda, "Expand 3x3 operation with ReLU (CUDA)");
}
"""

# Create a temporary directory for the CUDA extension
import tempfile
temp_dir = tempfile.mkdtemp()
with open(os.path.join(temp_dir, "fire_module_cuda.cpp"), "w") as f:
    f.write(cuda_source)

# Try to load the CUDA extension
try:
    fire_module_cuda = load(
        name="fire_module_cuda",
        sources=[os.path.join(temp_dir, "fire_module_cuda.cpp")],
        verbose=True
    )
    has_cuda_extension = True
except Exception as e:
    print(f"Failed to load CUDA extension: {e}")
    has_cuda_extension = False

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: Number of output classes
        """
        super(ModelNew, self).__init__()
        
        # Enable cuDNN autotuning
        torch.backends.cudnn.benchmark = True
        
        # Initial convolution layer with direct parameter access
        self.conv1_weight = nn.Parameter(torch.Tensor(96, 3, 7, 7))
        self.conv1_bias = nn.Parameter(torch.Tensor(96))
        
        # Fire module 1 parameters (in=96, squeeze=16, expand1x1=64, expand3x3=64)
        self.fire1_squeeze_weight = nn.Parameter(torch.Tensor(16, 96, 1, 1))
        self.fire1_squeeze_bias = nn.Parameter(torch.Tensor(16))
        self.fire1_expand1x1_weight = nn.Parameter(torch.Tensor(64, 16, 1, 1))
        self.fire1_expand1x1_bias = nn.Parameter(torch.Tensor(64))
        self.fire1_expand3x3_weight = nn.Parameter(torch.Tensor(64, 16, 3, 3))
        self.fire1_expand3x3_bias = nn.Parameter(torch.Tensor(64))
        
        # Fire module 2 parameters (in=128, squeeze=16, expand1x1=64, expand3x3=64)
        self.fire2_squeeze_weight = nn.Parameter(torch.Tensor(16, 128, 1, 1))
        self.fire2_squeeze_bias = nn.Parameter(torch.Tensor(16))
        self.fire2_expand1x1_weight = nn.Parameter(torch.Tensor(64, 16, 1, 1))
        self.fire2_expand1x1_bias = nn.Parameter(torch.Tensor(64))
        self.fire2_expand3x3_weight = nn.Parameter(torch.Tensor(64, 16, 3, 3))
        self.fire2_expand3x3_bias = nn.Parameter(torch.Tensor(64))
        
        # Fire module 3 parameters (in=128, squeeze=32, expand1x1=128, expand3x3=128)
        self.fire3_squeeze_weight = nn.Parameter(torch.Tensor(32, 128, 1, 1))
        self.fire3_squeeze_bias = nn.Parameter(torch.Tensor(32))
        self.fire3_expand1x1_weight = nn.Parameter(torch.Tensor(128, 32, 1, 1))
        self.fire3_expand1x1_bias = nn.Parameter(torch.Tensor(128))
        self.fire3_expand3x3_weight = nn.Parameter(torch.Tensor(128, 32, 3, 3))
        self.fire3_expand3x3_bias = nn.Parameter(torch.Tensor(128))
        
        # Fire module 4 parameters (in=256, squeeze=32, expand1x1=128, expand3x3=128)
        self.fire4_squeeze_weight = nn.Parameter(torch.Tensor(32, 256, 1, 1))
        self.fire4_squeeze_bias = nn.Parameter(torch.Tensor(32))
        self.fire4_expand1x1_weight = nn.Parameter(torch.Tensor(128, 32, 1, 1))
        self.fire4_expand1x1_bias = nn.Parameter(torch.Tensor(128))
        self.fire4_expand3x3_weight = nn.Parameter(torch.Tensor(128, 32, 3, 3))
        self.fire4_expand3x3_bias = nn.Parameter(torch.Tensor(128))
        
        # Fire module 5 parameters (in=256, squeeze=48, expand1x1=192, expand3x3=192)
        self.fire5_squeeze_weight = nn.Parameter(torch.Tensor(48, 256, 1, 1))
        self.fire5_squeeze_bias = nn.Parameter(torch.Tensor(48))
        self.fire5_expand1x1_weight = nn.Parameter(torch.Tensor(192, 48, 1, 1))
        self.fire5_expand1x1_bias = nn.Parameter(torch.Tensor(192))
        self.fire5_expand3x3_weight = nn.Parameter(torch.Tensor(192, 48, 3, 3))
        self.fire5_expand3x3_bias = nn.Parameter(torch.Tensor(192))
        
        # Fire module 6 parameters (in=384, squeeze=48, expand1x1=192, expand3x3=192)
        self.fire6_squeeze_weight = nn.Parameter(torch.Tensor(48, 384, 1, 1))
        self.fire6_squeeze_bias = nn.Parameter(torch.Tensor(48))
        self.fire6_expand1x1_weight = nn.Parameter(torch.Tensor(192, 48, 1, 1))
        self.fire6_expand1x1_bias = nn.Parameter(torch.Tensor(192))
        self.fire6_expand3x3_weight = nn.Parameter(torch.Tensor(192, 48, 3, 3))
        self.fire6_expand3x3_bias = nn.Parameter(torch.Tensor(192))
        
        # Fire module 7 parameters (in=384, squeeze=64, expand1x1=256, expand3x3=256)
        self.fire7_squeeze_weight = nn.Parameter(torch.Tensor(64, 384, 1, 1))
        self.fire7_squeeze_bias = nn.Parameter(torch.Tensor(64))
        self.fire7_expand1x1_weight = nn.Parameter(torch.Tensor(256, 64, 1, 1))
        self.fire7_expand1x1_bias = nn.Parameter(torch.Tensor(256))
        self.fire7_expand3x3_weight = nn.Parameter(torch.Tensor(256, 64, 3, 3))
        self.fire7_expand3x3_bias = nn.Parameter(torch.Tensor(256))
        
        # Fire module 8 parameters (in=512, squeeze=64, expand1x1=256, expand3x3=256)
        self.fire8_squeeze_weight = nn.Parameter(torch.Tensor(64, 512, 1, 1))
        self.fire8_squeeze_bias = nn.Parameter(torch.Tensor(64))
        self.fire8_expand1x1_weight = nn.Parameter(torch.Tensor(256, 64, 1, 1))
        self.fire8_expand1x1_bias = nn.Parameter(torch.Tensor(256))
        self.fire8_expand3x3_weight = nn.Parameter(torch.Tensor(256, 64, 3, 3))
        self.fire8_expand3x3_bias = nn.Parameter(torch.Tensor(256))
        
        # Classifier parameters
        self.classifier_weight = nn.Parameter(torch.Tensor(num_classes, 512, 1, 1))
        self.classifier_bias = nn.Parameter(torch.Tensor(num_classes))
        
        # Initialize all parameters
        self._initialize_weights()
        
        # Flag to determine if we can use the custom CUDA kernels
        self.use_cuda_kernels = has_cuda_extension
    
    def _initialize_weights(self):
        # Initialize conv1
        nn.init.kaiming_uniform_(self.conv1_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv1_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.conv1_bias, -bound, bound)
        
        # Initialize fire module parameters using a list for cleaner code
        fire_modules = [
            (self.fire1_squeeze_weight, self.fire1_squeeze_bias, self.fire1_expand1x1_weight, self.fire1_expand1x1_bias, self.fire1_expand3x3_weight, self.fire1_expand3x3_bias),
            (self.fire2_squeeze_weight, self.fire2_squeeze_bias, self.fire2_expand1x1_weight, self.fire2_expand1x1_bias, self.fire2_expand3x3_weight, self.fire2_expand3x3_bias),
            (self.fire3_squeeze_weight, self.fire3_squeeze_bias, self.fire3_expand1x1_weight, self.fire3_expand1x1_bias, self.fire3_expand3x3_weight, self.fire3_expand3x3_bias),
            (self.fire4_squeeze_weight, self.fire4_squeeze_bias, self.fire4_expand1x1_weight, self.fire4_expand1x1_bias, self.fire4_expand3x3_weight, self.fire4_expand3x3_bias),
            (self.fire5_squeeze_weight, self.fire5_squeeze_bias, self.fire5_expand1x1_weight, self.fire5_expand1x1_bias, self.fire5_expand3x3_weight, self.fire5_expand3x3_bias),
            (self.fire6_squeeze_weight, self.fire6_squeeze_bias, self.fire6_expand1x1_weight, self.fire6_expand1x1_bias, self.fire6_expand3x3_weight, self.fire6_expand3x3_bias),
            (self.fire7_squeeze_weight, self.fire7_squeeze_bias, self.fire7_expand1x1_weight, self.fire7_expand1x1_bias, self.fire7_expand3x3_weight, self.fire7_expand3x3_bias),
            (self.fire8_squeeze_weight, self.fire8_squeeze_bias, self.fire8_expand1x1_weight, self.fire8_expand1x1_bias, self.fire8_expand3x3_weight, self.fire8_expand3x3_bias),
        ]
        
        for squeeze_weight, squeeze_bias, expand1x1_weight, expand1x1_bias, expand3x3_weight, expand3x3_bias in fire_modules:
            # Squeeze weights and biases
            nn.init.kaiming_uniform_(squeeze_weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(squeeze_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(squeeze_bias, -bound, bound)
            
            # Expand 1x1 weights and biases
            nn.init.kaiming_uniform_(expand1x1_weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(expand1x1_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(expand1x1_bias, -bound, bound)
            
            # Expand 3x3 weights and biases
            nn.init.kaiming_uniform_(expand3x3_weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(expand3x3_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(expand3x3_bias, -bound, bound)
        
        # Initialize classifier
        nn.init.kaiming_uniform_(self.classifier_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.classifier_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.classifier_bias, -bound, bound)
    
    def _fire_forward_cuda(self, x, squeeze_weight, squeeze_bias, expand1x1_weight, expand1x1_bias, expand3x3_weight, expand3x3_bias):
        """
        Forward pass for a fire module using custom CUDA kernels
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Squeeze operation with ReLU using custom CUDA kernel
        squeeze_out = fire_module_cuda.squeeze_relu(x, squeeze_weight, squeeze_bias)
        
        # Expand operations using custom CUDA kernels
        expand1x1_out = fire_module_cuda.expand1x1_relu(squeeze_out, expand1x1_weight, expand1x1_bias)
        expand3x3_out = fire_module_cuda.expand3x3_relu(squeeze_out, expand3x3_weight, expand3x3_bias)
        
        # Concatenate results along channel dimension
        return torch.cat([expand1x1_out, expand3x3_out], 1)
    
    def _fire_forward_pytorch(self, x, squeeze_weight, squeeze_bias, expand1x1_weight, expand1x1_bias, expand3x3_weight, expand3x3_bias):
        """
        Optimized forward pass for a fire module using PyTorch operations
        """
        # Ensure input is contiguous
        x = x.contiguous()
        
        # Squeeze operation
        squeeze_out = F.conv2d(x, squeeze_weight, squeeze_bias)
        squeeze_out = F.relu(squeeze_out, inplace=True)
        
        # Expand operations - process both paths efficiently
        expand1x1_out = F.conv2d(squeeze_out, expand1x1_weight, expand1x1_bias)
        expand1x1_out = F.relu(expand1x1_out, inplace=True)
        
        expand3x3_out = F.conv2d(squeeze_out, expand3x3_weight, expand3x3_bias, padding=1)
        expand3x3_out = F.relu(expand3x3_out, inplace=True)
        
        # Concatenate results along channel dimension
        return torch.cat([expand1x1_out, expand3x3_out], 1)
    
    def _fire_forward(self, x, squeeze_weight, squeeze_bias, expand1x1_weight, expand1x1_bias, expand3x3_weight, expand3x3_bias):
        """
        Fire module forward pass that selects between CUDA and PyTorch implementations
        """
        if self.use_cuda_kernels and x.is_cuda:
            return self._fire_forward_cuda(x, squeeze_weight, squeeze_bias, expand1x1_weight, expand1x1_bias, expand3x3_weight, expand3x3_bias)
        else:
            return self._fire_forward_pytorch(x, squeeze_weight, squeeze_bias, expand1x1_weight, expand1x1_bias, expand3x3_weight, expand3x3_bias)
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        # Ensure input is contiguous for better memory access patterns
        x = x.contiguous()
        
        # Initial convolution with ReLU
        x = F.conv2d(x, self.conv1_weight, self.conv1_bias, stride=2)
        x = F.relu(x, inplace=True)
        
        # First maxpool
        x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)
        
        # Fire modules 1-3
        x = self._fire_forward(x, self.fire1_squeeze_weight, self.fire1_squeeze_bias, 
                              self.fire1_expand1x1_weight, self.fire1_expand1x1_bias, 
                              self.fire1_expand3x3_weight, self.fire1_expand3x3_bias)
        
        x = self._fire_forward(x, self.fire2_squeeze_weight, self.fire2_squeeze_bias, 
                              self.fire2_expand1x1_weight, self.fire2_expand1x1_bias, 
                              self.fire2_expand3x3_weight, self.fire2_expand3x3_bias)
        
        x = self._fire_forward(x, self.fire3_squeeze_weight, self.fire3_squeeze_bias, 
                              self.fire3_expand1x1_weight, self.fire3_expand1x1_bias, 
                              self.fire3_expand3x3_weight, self.fire3_expand3x3_bias)
        
        # Second maxpool
        x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)
        
        # Fire modules 4-7
        x = self._fire_forward(x, self.fire4_squeeze_weight, self.fire4_squeeze_bias, 
                              self.fire4_expand1x1_weight, self.fire4_expand1x1_bias, 
                              self.fire4_expand3x3_weight, self.fire4_expand3x3_bias)
        
        x = self._fire_forward(x, self.fire5_squeeze_weight, self.fire5_squeeze_bias, 
                              self.fire5_expand1x1_weight, self.fire5_expand1x1_bias, 
                              self.fire5_expand3x3_weight, self.fire5_expand3x3_bias)
        
        x = self._fire_forward(x, self.fire6_squeeze_weight, self.fire6_squeeze_bias, 
                              self.fire6_expand1x1_weight, self.fire6_expand1x1_bias, 
                              self.fire6_expand3x3_weight, self.fire6_expand3x3_bias)
        
        x = self._fire_forward(x, self.fire7_squeeze_weight, self.fire7_squeeze_bias, 
                              self.fire7_expand1x1_weight, self.fire7_expand1x1_bias, 
                              self.fire7_expand3x3_weight, self.fire7_expand3x3_bias)
        
        # Third maxpool
        x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)
        
        # Fire module 8
        x = self._fire_forward(x, self.fire8_squeeze_weight, self.fire8_squeeze_bias, 
                              self.fire8_expand1x1_weight, self.fire8_expand1x1_bias, 
                              self.fire8_expand3x3_weight, self.fire8_expand3x3_bias)
        
        # Classifier (no dropout since p=0.0)
        x = F.conv2d(x, self.classifier_weight, self.classifier_bias)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # Flatten output
        return torch.flatten(x, 1)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 1
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes]