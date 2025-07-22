import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedInceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        """
        Optimized Inception module with better memory access patterns
        """
        super(OptimizedInceptionModule, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch - separate for better parallelization
        self.branch3x3_reduce = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3 = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        
        # 5x5 convolution branch - separate for better parallelization
        self.branch5x5_reduce = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1)
        self.branch5x5 = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        
        # Max pooling branch
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_proj = nn.Conv2d(in_channels, pool_proj, kernel_size=1)
    
    def forward(self, x):
        """
        Optimized forward pass with parallel branch processing
        """
        # Process all initial operations in parallel
        branch1x1 = self.branch1x1(x)
        branch3x3_intermediate = self.branch3x3_reduce(x)
        branch5x5_intermediate = self.branch5x5_reduce(x)
        branch_pool_intermediate = self.branch_pool(x)
        
        # Process second stage operations
        branch3x3 = self.branch3x3(branch3x3_intermediate)
        branch5x5 = self.branch5x5(branch5x5_intermediate)
        branch_pool_proj = self.branch_pool_proj(branch_pool_intermediate)
        
        # Efficient concatenation
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool_proj]
        return torch.cat(outputs, 1)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Optimized GoogleNet Inception V1 implementation
        
        Args:
            num_classes: Number of output classes
        """
        super(ModelNew, self).__init__()
        
        # Enable cuDNN benchmarking for optimized convolution performance
        torch.backends.cudnn.benchmark = True
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Inception modules
        self.inception3a = OptimizedInceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = OptimizedInceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception4a = OptimizedInceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = OptimizedInceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = OptimizedInceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = OptimizedInceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = OptimizedInceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception5a = OptimizedInceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = OptimizedInceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1024, num_classes)
        
        # Apply memory format optimization
        self._optimize_memory_format()
    
    def _optimize_memory_format(self):
        """Convert model parameters to channels_last memory format"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convert input to channels_last memory format if on CUDA
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        
        # Initial layers with ReLU activations
        x = F.relu(self.conv1(x), inplace=True)
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = self.maxpool2(x)
        
        # Inception modules
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes]