import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation with optimizations.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(ModelNew, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.hidden_dim = in_channels * expand_ratio
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        
        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.expand_bn = nn.BatchNorm2d(self.hidden_dim)
            # Pre-compute folded weights and biases for expansion
            self.register_buffer('expand_folded_weight', torch.zeros_like(self.expand_conv.weight))
            self.register_buffer('expand_folded_bias', torch.zeros(self.hidden_dim, device=self.expand_conv.weight.device))
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, 
                                       stride=stride, padding=self.padding, groups=self.hidden_dim, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(self.hidden_dim)
        # Pre-compute folded weights and biases for depthwise
        self.register_buffer('depthwise_folded_weight', torch.zeros_like(self.depthwise_conv.weight))
        self.register_buffer('depthwise_folded_bias', torch.zeros(self.hidden_dim, device=self.depthwise_conv.weight.device))
        
        # Projection phase
        self.project_conv = nn.Conv2d(self.hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        # Pre-compute folded weights and biases for projection
        self.register_buffer('project_folded_weight', torch.zeros_like(self.project_conv.weight))
        self.register_buffer('project_folded_bias', torch.zeros(out_channels, device=self.project_conv.weight.device))
        
        # Flag to indicate if weights are folded
        self.weights_folded = False
        
        # Create CUDA stream for optimized execution
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
    
    def _fold_bn_into_conv(self):
        """Fold BatchNorm parameters into Conv weights and biases"""
        if self.weights_folded:
            return
        
        with torch.no_grad():
            # Fold expansion phase if it exists
            if self.expand_ratio != 1:
                # Compute BN scaling factors
                bn_var = self.expand_bn.running_var
                bn_eps = self.expand_bn.eps
                bn_std_inv = torch.rsqrt(bn_var + bn_eps)
                bn_weight = self.expand_bn.weight
                bn_bias = self.expand_bn.bias
                bn_mean = self.expand_bn.running_mean
                
                # Fold parameters
                weight_scale = bn_weight * bn_std_inv
                self.expand_folded_weight.copy_(self.expand_conv.weight * weight_scale.view(-1, 1, 1, 1))
                self.expand_folded_bias.copy_(bn_bias - bn_mean * weight_scale)
            
            # Fold depthwise phase
            bn_var = self.depthwise_bn.running_var
            bn_eps = self.depthwise_bn.eps
            bn_std_inv = torch.rsqrt(bn_var + bn_eps)
            bn_weight = self.depthwise_bn.weight
            bn_bias = self.depthwise_bn.bias
            bn_mean = self.depthwise_bn.running_mean
            
            # Fold parameters
            weight_scale = bn_weight * bn_std_inv
            self.depthwise_folded_weight.copy_(self.depthwise_conv.weight * weight_scale.view(-1, 1, 1, 1))
            self.depthwise_folded_bias.copy_(bn_bias - bn_mean * weight_scale)
            
            # Fold projection phase
            bn_var = self.project_bn.running_var
            bn_eps = self.project_bn.eps
            bn_std_inv = torch.rsqrt(bn_var + bn_eps)
            bn_weight = self.project_bn.weight
            bn_bias = self.project_bn.bias
            bn_mean = self.project_bn.running_mean
            
            # Fold parameters
            weight_scale = bn_weight * bn_std_inv
            self.project_folded_weight.copy_(self.project_conv.weight * weight_scale.view(-1, 1, 1, 1))
            self.project_folded_bias.copy_(bn_bias - bn_mean * weight_scale)
            
        self.weights_folded = True
    
    def forward(self, x):
        """
        Forward pass of the MBConv block.

        :param x: The input tensor, shape (batch_size, in_channels, H, W)
        :return: The output tensor, shape (batch_size, out_channels, H', W')
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Store input for residual connection
        identity = x if self.use_residual else None
        
        # Fold BN parameters into conv weights and biases if not done already
        if not self.weights_folded:
            self._fold_bn_into_conv()
        
        # Optimized forward pass with minimal overhead
        if x.is_cuda and self.stream is not None:
            with torch.cuda.stream(self.stream):
                # Expansion phase with fused operations
                if self.expand_ratio != 1:
                    x = F.conv2d(x, self.expand_folded_weight, self.expand_folded_bias, stride=1, padding=0)
                    x.clamp_(min=0.0, max=6.0)  # In-place ReLU6 for memory efficiency
                
                # Depthwise convolution phase with fused operations
                x = F.conv2d(x, self.depthwise_folded_weight, self.depthwise_folded_bias, 
                           stride=self.stride, padding=self.padding, groups=self.hidden_dim)
                x.clamp_(min=0.0, max=6.0)  # In-place ReLU6 for memory efficiency
                
                # Projection phase
                x = F.conv2d(x, self.project_folded_weight, self.project_folded_bias, stride=1, padding=0)
                
                # Apply residual connection if needed
                if self.use_residual:
                    x.add_(identity)  # In-place addition for memory efficiency
        else:
            # CPU or fallback path
            # Expansion phase
            if self.expand_ratio != 1:
                x = F.conv2d(x, self.expand_folded_weight, self.expand_folded_bias, stride=1, padding=0)
                x = torch.clamp(x, min=0.0, max=6.0)
            
            # Depthwise convolution phase
            x = F.conv2d(x, self.depthwise_folded_weight, self.depthwise_folded_bias, 
                       stride=self.stride, padding=self.padding, groups=self.hidden_dim)
            x = torch.clamp(x, min=0.0, max=6.0)
            
            # Projection phase
            x = F.conv2d(x, self.project_folded_weight, self.project_folded_bias, stride=1, padding=0)
            
            # Apply residual connection if needed
            if self.use_residual:
                x = x + identity
        
        return x

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
in_channels = 112
out_channels = 192
kernel_size = 5
stride = 2
expand_ratio = 6

def get_inputs():
    return [torch.randn(batch_size, in_channels, 224, 224)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, expand_ratio]