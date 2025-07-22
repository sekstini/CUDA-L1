import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        """
        :param in_channels: Number of input channels
        :param out_1x1: Number of output channels for the 1x1 convolution
        :param reduce_3x3: Number of output channels for the 1x1 reduction before 3x3 convolution
        :param out_3x3: Number of output channels for the 3x3 convolution
        :param reduce_5x5: Number of output channels for the 1x1 reduction before 5x5 convolution
        :param out_5x5: Number of output channels for the 5x5 convolution
        :param pool_proj: Number of output channels for the pooling projection
        """
        super(ModelNew, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3_reduce = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3 = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        
        # 5x5 convolution branch
        self.branch5x5_reduce = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1)
        self.branch5x5 = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        
        # Max pooling branch
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_proj = nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        
        # Pre-compute output channel indices for efficient slicing
        self.idx1 = out_1x1
        self.idx2 = out_1x1 + out_3x3
        self.idx3 = out_1x1 + out_3x3 + out_5x5
        self.total_out_channels = out_1x1 + out_3x3 + out_5x5 + pool_proj
        
        # Initialize CUDA streams for parallel execution
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.stream1 = torch.cuda.Stream()
            self.stream2 = torch.cuda.Stream()
            self.stream3 = torch.cuda.Stream()
            self.stream4 = torch.cuda.Stream()
        
        # Perform warmup to ensure kernels are compiled
        if self.use_cuda:
            self._warmup()
    
    def _warmup(self):
        """Pre-compile kernels for common sizes to reduce JIT compilation overhead"""
        try:
            with torch.no_grad():
                # Create a small dummy input to warm up the kernels
                dummy_input = torch.zeros(1, in_channels, 32, 32, device='cuda')
                self.forward(dummy_input)
                torch.cuda.synchronize()
        except:
            # If warmup fails for any reason, just continue
            pass
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        batch_size, _, height, width = x.shape
        
        # Pre-allocate output tensor with the correct shape and memory format
        output = torch.empty(
            (batch_size, self.total_out_channels, height, width),
            dtype=x.dtype, device=x.device,
            memory_format=torch.channels_last if x.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
        )
        
        if self.use_cuda and x.is_cuda:
            # Record the current stream to restore it later
            current_stream = torch.cuda.current_stream()
            
            # Process branch 1 (1x1 convolution) in stream1
            with torch.cuda.stream(self.stream1):
                branch1x1_out = self.branch1x1(x)
                output[:, :self.idx1] = branch1x1_out
            
            # Process branch 2 (3x3 convolution) in stream2
            with torch.cuda.stream(self.stream2):
                branch3x3_reduced = self.branch3x3_reduce(x)
                branch3x3_out = self.branch3x3(branch3x3_reduced)
                output[:, self.idx1:self.idx2] = branch3x3_out
            
            # Process branch 3 (5x5 convolution) in stream3
            with torch.cuda.stream(self.stream3):
                branch5x5_reduced = self.branch5x5_reduce(x)
                branch5x5_out = self.branch5x5(branch5x5_reduced)
                output[:, self.idx2:self.idx3] = branch5x5_out
            
            # Process branch 4 (pool projection) in stream4
            with torch.cuda.stream(self.stream4):
                branch_pool = self.maxpool(x)
                branch_pool_out = self.branch_pool_proj(branch_pool)
                output[:, self.idx3:] = branch_pool_out
            
            # Synchronize all streams with the current stream
            current_stream.synchronize()
            
        else:
            # Fallback to sequential execution for CPU or when CUDA is not available
            branch1x1_out = self.branch1x1(x)
            output[:, :self.idx1] = branch1x1_out
            
            branch3x3_reduced = self.branch3x3_reduce(x)
            branch3x3_out = self.branch3x3(branch3x3_reduced)
            output[:, self.idx1:self.idx2] = branch3x3_out
            
            branch5x5_reduced = self.branch5x5_reduce(x)
            branch5x5_out = self.branch5x5(branch5x5_reduced)
            output[:, self.idx2:self.idx3] = branch5x5_out
            
            branch_pool = self.maxpool(x)
            branch_pool_out = self.branch_pool_proj(branch_pool)
            output[:, self.idx3:] = branch_pool_out
        
        return output

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
in_channels = 480
out_1x1 = 192
reduce_3x3 = 96
out_3x3 = 208
reduce_5x5 = 16
out_5x5 = 48
pool_proj = 64
batch_size = 10
height = 224
width = 224

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj]