import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features  
        bias_shape (tuple): Shape of the bias tensor
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        # Use the exact same structure as reference implementation
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Pre-transpose weight for optimized matrix multiplication
        with torch.no_grad():
            self.register_buffer('weight_t', self.gemm.weight.t().contiguous())
        
        # Track weight version to minimize unnecessary updates
        self._weight_version = 0
        
        # Register forward pre-hook to update transposed weight before forward execution
        self.gemm.register_forward_pre_hook(self._update_weight_t)
    
    def _update_weight_t(self, module, input):
        """
        Update the transposed weight if the original weight has changed
        This is called automatically before each forward pass
        """
        # Simplified version tracking for better performance
        current_version = getattr(self.gemm.weight, '_version', self._weight_version + 1)
        if current_version != self._weight_version:
            with torch.no_grad():
                self.weight_t.copy_(self.gemm.weight.t().contiguous())
                self._weight_version = current_version
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Only ensure contiguity if necessary to avoid redundant operations
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Fused matrix multiplication and bias addition using addmm
        # torch.addmm: out = beta * input + alpha * (mat1 @ mat2)
        # Here beta=1, input=bias, alpha=1, mat1=x, mat2=weight_t
        output = torch.addmm(self.bias, x, self.weight_t)
        
        # In-place ReLU to avoid additional memory allocation
        output.relu_()
        
        return output

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 1024
out_features = 512
bias_shape = (out_features,)

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation  
    return [in_features, out_features, bias_shape]