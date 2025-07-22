import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias (bool): Whether to use bias
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias similar to nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters using same method as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Cache for transposed weight - simple attribute instead of buffer
        self._weight_t = None
        self._weight_version = None
        
        # Pre-compute constants as simple tensors for reduced overhead
        self._half = torch.tensor(0.5)
        self._neg_one = torch.tensor(-1.0)
        self._pos_one = torch.tensor(1.0)
        self._device_initialized = None
    
    def forward(self, x):
        """
        Optimized forward pass with maximum memory efficiency
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Fast path for contiguous tensors
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Get device of input tensor
        device = x.device
        
        # Move constants to same device as input if needed - use simple comparison
        if self._device_initialized != device:
            # Move all constants at once to minimize device transfers
            self._half = self._half.to(device)
            self._neg_one = self._neg_one.to(device)
            self._pos_one = self._pos_one.to(device)
            self._device_initialized = device
        
        # Cache constants locally to avoid attribute lookup overhead
        half = self._half
        neg_one = self._neg_one
        pos_one = self._pos_one
        
        # Optimize weight transposition check
        weight = self.weight
        weight_version = weight._version
        
        # Lazily initialize or update transposed weight matrix - only when needed
        if self._weight_t is None or self._weight_version != weight_version:
            self._weight_t = weight.t().contiguous()
            self._weight_version = weight_version
        
        # Cache transposed weight locally
        weight_t = self._weight_t
        
        # Linear transformation (GEMM) - use addmm for better performance
        if self.bias is not None:
            output = torch.addmm(self.bias, x, weight_t)
        else:
            output = torch.mm(x, weight_t)
        
        # Use PyTorch's optimized SiLU (Swish) activation function
        # SiLU(x) = x * sigmoid(x) which is exactly Swish
        output = torch.nn.functional.silu(output)
        
        # Division by 2.0 (using multiplication by 0.5 for better performance)
        output.mul_(half)
        
        # First clamp between -1.0 and 1.0 (in-place)
        output.clamp_(neg_one, pos_one)
        
        # Tanh activation - use in-place version
        output.tanh_()
        
        # Final clamp between -1.0 and 1.0 (in-place)
        # Note: This is technically redundant since tanh output is already in [-1,1]
        # but keeping for exact functional equivalence
        output.clamp_(neg_one, pos_one)
        
        return output

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_features, out_features]