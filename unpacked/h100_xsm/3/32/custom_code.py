import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

# Define CUDA kernel for fused convolution reshape and linear projection
cuda_source = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void fused_reshape_linear_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int embed_dim,
    int flatten_size) {
    
    // Each thread handles one output element
    int b = blockIdx.x;
    int e = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (b < batch_size && e < embed_dim) {
        scalar_t sum = 0;
        
        // Compute dot product between reshaped input and weight
        for (int i = 0; i < flatten_size; ++i) {
            sum += input[b * flatten_size + i] * weight[e * flatten_size + i];
        }
        
        // Add bias if provided
        if (bias != nullptr) {
            sum += bias[e];
        }
        
        output[b * embed_dim + e] = sum;
    }
}

// Forward function for the fused operation
std::vector<torch::Tensor> fused_reshape_linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto flatten_size = input.size(1) * input.size(2) * input.size(3);
    auto embed_dim = weight.size(0);
    
    // Reshape input for processing
    auto input_reshaped = input.reshape({batch_size, flatten_size});
    
    // Create output tensor
    auto output = torch::empty({batch_size, embed_dim}, input.options());
    
    // Calculate grid and block dimensions
    const int threads = 256;
    const dim3 blocks((batch_size + 1 - 1) / 1, (embed_dim + threads - 1) / threads);
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_reshape_linear_kernel", ([&] {
        fused_reshape_linear_kernel<scalar_t><<<blocks, threads>>>(
            input_reshaped.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            embed_dim,
            flatten_size);
    }));
    
    return {output};
}

// Backward function (simplified for now)
std::vector<torch::Tensor> fused_reshape_linear_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {
    
    auto batch_size = input.size(0);
    auto flatten_size = input.size(1) * input.size(2) * input.size(3);
    
    // Reshape input for gradient calculation
    auto input_reshaped = input.reshape({batch_size, flatten_size});
    
    // Calculate gradients using PyTorch operations for now
    auto grad_input = torch::matmul(grad_output, weight).reshape_as(input);
    auto grad_weight = torch::matmul(grad_output.transpose(0, 1), input_reshaped);
    
    torch::Tensor grad_bias;
    if (bias.defined()) {
        grad_bias = grad_output.sum(0);
    } else {
        grad_bias = torch::Tensor();
    }
    
    return {grad_input, grad_weight, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_reshape_linear_forward, "Fused reshape and linear forward");
    m.def("backward", &fused_reshape_linear_backward, "Fused reshape and linear backward");
}
'''

# Define custom autograd function for the fused operation
class FusedReshapeLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = fused_ops.forward(input, weight, bias)[0]
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_ops.backward(grad_output, input, weight, bias)
        return grad_input, grad_weight, grad_bias if bias is not None else None

# Define custom module for the fused operation
class FusedReshapeLinearModule(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(FusedReshapeLinearModule, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        return FusedReshapeLinear.apply(input, self.weight, self.bias)

# Only compile the extension if we can
try:
    fused_ops = load_inline(
        name='fused_ops',
        cpp_sources=[],
        cuda_sources=[cuda_source],
        functions=['forward', 'backward'],
        with_cuda=True,
        verbose=True
    )
    has_cuda_extension = True
except Exception as e:
    print(f"Warning: Could not load CUDA extension: {e}")
    has_cuda_extension = False

class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3):
        """
        Convolutional Vision Transformer (CViT) implementation.
        :param num_classes: Number of output classes for classification.
        :param embed_dim: Dimensionality of the embedding space.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of transformer layers.
        :param mlp_ratio: Ratio of the MLP hidden dimension to the embedding dimension.
        :param patch_size: Size of the convolutional patches.
        :param in_channels: Number of input channels (e.g., 3 for RGB images).
        """
        super(ModelNew, self).__init__()

        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Calculate spatial dimensions after convolution
        self.spatial_size = 32 // patch_size
        self.flatten_size = embed_dim * self.spatial_size * self.spatial_size
        
        # Use custom fused operation if available, otherwise fallback to standard PyTorch
        if has_cuda_extension:
            self.fused_proj = FusedReshapeLinearModule(self.flatten_size, embed_dim)
        else:
            self.linear_proj = nn.Linear(self.flatten_size, embed_dim)

        # Create transformer layers
        transformer_layers = []
        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio), 
                dropout=0.0
            )
            transformer_layers.append(layer)
        
        # JIT script the transformer layers for optimization
        self.transformer_layers = torch.jit.script(nn.Sequential(*transformer_layers))
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)
        
        # Pre-compute cls token expansion for common batch sizes
        self._cached_cls_tokens = {}

    def _get_cls_tokens(self, batch_size):
        """Get cached cls tokens for the given batch size"""
        if batch_size not in self._cached_cls_tokens:
            self._cached_cls_tokens[batch_size] = self.cls_token.expand(batch_size, -1, -1)
        return self._cached_cls_tokens[batch_size]

    def forward(self, x):
        """
        Forward pass of the CViT model.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of shape (B, num_classes)
        """
        B = x.shape[0]
        
        # Process patches with convolution
        x = self.conv1(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        
        # Apply fused reshape + linear projection if available, otherwise use standard operations
        if has_cuda_extension:
            x = self.fused_proj(x)  # (B, embed_dim)
        else:
            x = x.reshape(B, self.flatten_size)
            x = self.linear_proj(x)  # (B, embed_dim)
        
        # Add cls token using cached version
        cls_tokens = self._get_cls_tokens(B)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)  # (B, 1+N, embed_dim)

        # Apply transformer layers with JIT optimization
        x = self.transformer_layers(x)

        # Classify based on cls token
        x = x[:, 0]  # Get the cls token's output
        x = self.fc_out(x)  # (B, num_classes)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
image_size = 32
embed_dim = 128
in_channels = 3
num_heads = 4
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, in_channels, image_size, image_size)]

def get_init_inputs():
    return [num_classes, embed_dim, num_heads]