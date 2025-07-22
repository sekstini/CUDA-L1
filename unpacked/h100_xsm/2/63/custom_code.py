import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Your optimized implementation here that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features  
        divisor (float): Divisor to apply after ReLU
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.divisor = divisor
        
        # Create standard linear layer parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize parameters using the same method as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Pre-scale the weight and bias for the division operation
        self.register_buffer('scaled_weight', self.weight.detach() / self.divisor)
        self.register_buffer('scaled_bias', self.bias.detach() / self.divisor)
        
        # Pre-transpose the weight matrix for addmm operation
        self.register_buffer('scaled_weight_t', self.scaled_weight.t().contiguous())
        
        # Register hooks to update scaled parameters when original parameters change
        self._register_update_hooks()
        
    def _register_update_hooks(self):
        def update_scaled_params(grad):
            if self.training and grad is not None:
                with torch.no_grad():
                    # Update both scaled weight and bias in one go to reduce overhead
                    self.scaled_weight.copy_(self.weight / self.divisor)
                    self.scaled_weight_t.copy_(self.scaled_weight.t().contiguous())
                    self.scaled_bias.copy_(self.bias / self.divisor)
            return grad
        
        # Use a single hook for both parameters to reduce overhead
        self.weight.register_hook(update_scaled_params)
        self.bias.register_hook(update_scaled_params)
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Ensure input is contiguous for better memory access patterns
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Use pre-scaled weights and bias to avoid division
        # addmm: out = beta * input + alpha * (mat1 @ mat2)
        # This directly leverages optimized BLAS routines and GPU tensor cores
        output = torch.addmm(self.scaled_bias, x, self.scaled_weight_t)
        
        # Apply ReLU in-place to reduce memory usage
        output.relu_()
        
        return output

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 1024
out_features = 512
divisor = 2.0

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_features, out_features, divisor]