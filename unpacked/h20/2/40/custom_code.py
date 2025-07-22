import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Your optimized implementation here that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features  
        scaling_factor (float): Scaling factor to apply
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        
        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Calculate the effective scaling factor
        effective_scaling = 1.0 + scaling_factor
        
        # Create parameters with optimal initialization
        weight = torch.empty(out_features, in_features, dtype=torch.float32)
        bias = torch.empty(out_features, dtype=torch.float32)
        
        # Initialize parameters the same way nn.Linear does
        nn.init.kaiming_uniform_(weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(bias, -bound, bound)
        
        # Pre-scale the weights and bias by the effective scaling factor
        weight.mul_(effective_scaling)
        bias.mul_(effective_scaling)
        
        # Pre-transpose the weight matrix for torch.addmm
        self.weight_t = nn.Parameter(weight.t().contiguous())
        self.bias = nn.Parameter(bias.contiguous())
        
        # Store original parameters for reference
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Use torch.addmm which combines matrix multiplication and bias addition
        # in one highly optimized operation - no conditional checks for maximum performance
        return torch.addmm(self.bias, x, self.weight_t)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 64
out_features = 128
scaling_factor = 0.5

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation  
    return [in_features, out_features, scaling_factor]