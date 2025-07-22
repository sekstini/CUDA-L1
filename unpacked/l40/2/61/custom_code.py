import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed 3D convolution, applies ReLU, 
    and then applies group normalization.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the convolving kernel
        groups (int): Number of groups for GroupNorm
        bias (bool): If True, adds a learnable bias to the output
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.groups = groups
        self.bias = bias
        
        # Enable cuDNN benchmarking for optimal algorithm selection
        torch.backends.cudnn.benchmark = True
        
        # Create weight parameter with correct dimensions for ConvTranspose3d
        # For ConvTranspose3d, weight shape is (in_channels, out_channels, kernel_d, kernel_h, kernel_w)
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, *self.kernel_size))
        
        # Initialize weights using the same method as nn.ConvTranspose3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias_param = nn.Parameter(torch.Tensor(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)
        else:
            self.register_parameter('bias_param', None)
        
        # For GroupNorm, we need gamma and beta parameters
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        
        # Create standard PyTorch modules as fallback
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, self.kernel_size, bias=bias)
        self.relu = nn.ReLU(inplace=True)  # Use in-place ReLU to save memory
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        
        # Copy our initialized weights to the standard module
        with torch.no_grad():
            self.conv_transpose.weight.copy_(self.weight)
            if bias and self.bias_param is not None:
                self.conv_transpose.bias.copy_(self.bias_param)
        
        # Pre-allocate stream for asynchronous operations
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
    
    def _optimized_forward(self, x):
        """
        Optimized forward pass implementation using PyTorch's functional API
        """
        # Step 1: ConvTranspose3d using F.conv_transpose3d with our parameters
        out = F.conv_transpose3d(
            x, 
            self.weight, 
            self.bias_param, 
            stride=1, 
            padding=0
        )
        
        # Step 2: ReLU (in-place to save memory)
        out.relu_()
        
        # Step 3: GroupNorm using F.group_norm with our parameters
        out = F.group_norm(
            out, 
            num_groups=self.groups, 
            weight=self.gamma, 
            bias=self.beta, 
            eps=1e-5
        )
        
        return out
    
    def _fallback_forward(self, x):
        """
        Fallback implementation using standard PyTorch modules
        """
        x = self.conv_transpose(x)
        x = self.relu(x)
        x = self.group_norm(x)
        return x
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        if x.is_cuda:
            try:
                # Try the optimized implementation first
                with torch.cuda.stream(self.stream):
                    return self._optimized_forward(x)
            except Exception:
                # Fall back to the standard implementation if there's an error
                return self._fallback_forward(x)
        else:
            # Use the standard implementation for CPU tensors
            return self._fallback_forward(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 64
out_channels = 128
D, H, W = 8, 16, 16
kernel_size = 3
groups = 8
bias = False

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups, bias]