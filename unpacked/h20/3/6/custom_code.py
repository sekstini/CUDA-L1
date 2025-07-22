import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class InceptionForward(Function):
    @staticmethod
    def forward(ctx, x, branch1x1, branch3x3_reduce, branch3x3, branch5x5_reduce, branch5x5, branch_pool_proj):
        # Save for backward
        ctx.save_for_backward(x, branch1x1.weight, branch1x1.bias, 
                             branch3x3_reduce.weight, branch3x3_reduce.bias,
                             branch3x3.weight, branch3x3.bias,
                             branch5x5_reduce.weight, branch5x5_reduce.bias,
                             branch5x5.weight, branch5x5.bias,
                             branch_pool_proj.weight, branch_pool_proj.bias)
        
        # Use standard PyTorch operations but in an optimized way
        # Branch 1: 1x1 convolution
        branch1x1_out = F.conv2d(x, branch1x1.weight, branch1x1.bias)
        
        # Branch 2: 1x1 reduction followed by 3x3 convolution
        branch3x3_reduce_out = F.conv2d(x, branch3x3_reduce.weight, branch3x3_reduce.bias)
        branch3x3_out = F.conv2d(branch3x3_reduce_out, branch3x3.weight, branch3x3.bias, padding=1)
        
        # Branch 3: 1x1 reduction followed by 5x5 convolution
        branch5x5_reduce_out = F.conv2d(x, branch5x5_reduce.weight, branch5x5_reduce.bias)
        branch5x5_out = F.conv2d(branch5x5_reduce_out, branch5x5.weight, branch5x5.bias, padding=2)
        
        # Branch 4: MaxPool followed by 1x1 convolution
        branch_pool_out = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool_proj_out = F.conv2d(branch_pool_out, branch_pool_proj.weight, branch_pool_proj.bias)
        
        # Concatenate the outputs along the channel dimension
        return torch.cat([branch1x1_out, branch3x3_out, branch5x5_out, branch_pool_proj_out], 1)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, w1x1, b1x1, w3x3_reduce, b3x3_reduce, w3x3, b3x3, \
        w5x5_reduce, b5x5_reduce, w5x5, b5x5, w_pool, b_pool = ctx.saved_tensors
        
        # Get dimensions for splitting the gradient
        out_1x1 = w1x1.size(0)
        out_3x3 = w3x3.size(0)
        out_5x5 = w5x5.size(0)
        
        # Split the gradient
        grad_branch1x1 = grad_output[:, :out_1x1]
        grad_branch3x3 = grad_output[:, out_1x1:out_1x1+out_3x3]
        grad_branch5x5 = grad_output[:, out_1x1+out_3x3:out_1x1+out_3x3+out_5x5]
        grad_branch_pool = grad_output[:, out_1x1+out_3x3+out_5x5:]
        
        # Compute gradients using PyTorch autograd
        grad_x = torch.zeros_like(x)
        
        # Branch 1: 1x1 convolution
        x_1 = x.detach().requires_grad_()
        branch1x1_out = F.conv2d(x_1, w1x1, b1x1)
        branch1x1_out.backward(grad_branch1x1)
        grad_x += x_1.grad
        
        # Branch 2: 1x1 reduction followed by 3x3 convolution
        x_2 = x.detach().requires_grad_()
        branch3x3_reduce_out = F.conv2d(x_2, w3x3_reduce, b3x3_reduce)
        branch3x3_out = F.conv2d(branch3x3_reduce_out, w3x3, b3x3, padding=1)
        branch3x3_out.backward(grad_branch3x3)
        grad_x += x_2.grad
        
        # Branch 3: 1x1 reduction followed by 5x5 convolution
        x_3 = x.detach().requires_grad_()
        branch5x5_reduce_out = F.conv2d(x_3, w5x5_reduce, b5x5_reduce)
        branch5x5_out = F.conv2d(branch5x5_reduce_out, w5x5, b5x5, padding=2)
        branch5x5_out.backward(grad_branch5x5)
        grad_x += x_3.grad
        
        # Branch 4: MaxPool followed by 1x1 convolution
        x_4 = x.detach().requires_grad_()
        branch_pool_out = F.max_pool2d(x_4, kernel_size=3, stride=1, padding=1)
        branch_pool_proj_out = F.conv2d(branch_pool_out, w_pool, b_pool)
        branch_pool_proj_out.backward(grad_branch_pool)
        grad_x += x_4.grad
        
        return grad_x, None, None, None, None, None, None

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
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
        
        # Store the individual components for optimized forward pass
        self.branch1x1_conv = self.branch1x1
        self.branch3x3_reduce = self.branch3x3[0]
        self.branch3x3_conv = self.branch3x3[1]
        self.branch5x5_reduce = self.branch5x5[0]
        self.branch5x5_conv = self.branch5x5[1]
        self.branch_pool_proj = self.branch_pool[1]
        
        # Flag to control whether to use optimized forward
        self.use_optimized = True
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        if self.use_optimized and x.is_cuda:
            return InceptionForward.apply(
                x, 
                self.branch1x1_conv,
                self.branch3x3_reduce, 
                self.branch3x3_conv,
                self.branch5x5_reduce, 
                self.branch5x5_conv,
                self.branch_pool_proj
            )
        else:
            # Standard implementation using PyTorch modules
            branch1x1 = self.branch1x1(x)
            branch3x3 = self.branch3x3(x)
            branch5x5 = self.branch5x5(x)
            branch_pool = self.branch_pool(x)
            
            outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
            return torch.cat(outputs, 1)

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