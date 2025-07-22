import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvReLUMaxPool(nn.Module):
    """
    Fused Conv2d + ReLU + MaxPool2d module for better performance
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pool_kernel_size=3, pool_stride=2):
        super(ConvReLUMaxPool, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        
    def forward(self, x):
        x = self.conv(x)
        x.relu_()  # In-place ReLU for better performance
        return F.max_pool2d(x, self.pool_kernel_size, self.pool_stride)

class ConvReLU(nn.Module):
    """
    Fused Conv2d + ReLU module for better performance
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        x = self.conv(x)
        x.relu_()  # In-place ReLU for better performance
        return x

class OptimizedLinearReLU(nn.Module):
    """
    Optimized Linear + ReLU module using torch.addmm for efficient matrix multiplication
    """
    def __init__(self, in_features, out_features):
        super(OptimizedLinearReLU, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        # Use torch.addmm for fused matrix multiply-add
        output = torch.addmm(self.bias, x, self.weight.t())
        output.clamp_(min=0)  # In-place ReLU using clamp_
        return output

class OptimizedLinear(nn.Module):
    """
    Optimized Linear module using torch.addmm for efficient matrix multiplication
    """
    def __init__(self, in_features, out_features):
        super(OptimizedLinear, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        # Use torch.addmm for fused matrix multiply-add
        return torch.addmm(self.bias, x, self.weight.t())

class FirstConvReLUMaxPool(nn.Module):
    """
    Specialized first convolutional layer with ReLU and MaxPool
    This is a critical layer as it processes the largest input tensor
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pool_kernel_size=3, pool_stride=2):
        super(FirstConvReLUMaxPool, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.stride = stride
        self.padding = padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        # Optimized convolution using F.conv2d directly
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        x.relu_()  # In-place ReLU
        return F.max_pool2d(x, self.pool_kernel_size, self.pool_stride)

class ConvToFC(nn.Module):
    """
    Optimized transition from convolutional to fully connected layers
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pool_kernel_size=3, pool_stride=2, fc_size=4096):
        super(ConvToFC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.fc = OptimizedLinearReLU(out_channels * 6 * 6, fc_size)
        
    def forward(self, x):
        x = self.conv(x)
        x.relu_()  # In-place ReLU
        x = F.max_pool2d(x, self.pool_kernel_size, self.pool_stride)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.fc(x)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(ModelNew, self).__init__()
        
        # First block: Conv + ReLU + MaxPool (specialized implementation)
        self.block1 = FirstConvReLUMaxPool(
            in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2,
            pool_kernel_size=3, pool_stride=2
        )
        
        # Second block: Conv + ReLU + MaxPool
        self.block2 = ConvReLUMaxPool(
            in_channels=96, out_channels=256, kernel_size=5, padding=2,
            pool_kernel_size=3, pool_stride=2
        )
        
        # Third block: Conv + ReLU
        self.block3 = ConvReLU(
            in_channels=256, out_channels=384, kernel_size=3, padding=1
        )
        
        # Fourth block: Conv + ReLU
        self.block4 = ConvReLU(
            in_channels=384, out_channels=384, kernel_size=3, padding=1
        )
        
        # Fifth block: Conv + ReLU + MaxPool with transition to FC
        self.block5 = ConvToFC(
            in_channels=384, out_channels=256, kernel_size=3, padding=1,
            pool_kernel_size=3, pool_stride=2, fc_size=4096
        )
        
        # Remaining fully connected layers
        self.fc2 = OptimizedLinearReLU(4096, 4096)
        self.fc3 = OptimizedLinear(4096, num_classes)
        
        # Create CUDA stream for efficient execution
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # Ensure input tensor is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Use CUDA stream for efficient execution
        if x.is_cuda:
            with torch.cuda.stream(self.stream):
                # Convolutional layers with fused operations
                x = self.block1(x)
                x = self.block2(x)
                x = self.block3(x)
                x = self.block4(x)
                
                # Optimized transition from conv to fc layers
                x = self.block5(x)
                
                # Remaining fully connected layers
                x = self.fc2(x)
                x = self.fc3(x)
        else:
            # CPU execution path
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.fc2(x)
            x = self.fc3(x)
        
        return x

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]