import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # Create a temporary linear layer to get weights and biases
        linear = nn.Linear(in_features, out_features)
        
        # Precompute the sum of weights and bias during initialization
        with torch.no_grad():
            # Sum the weights across the output dimension and reshape for matmul
            weight_sum = torch.sum(linear.weight, dim=0).contiguous()
            self.weight_sum = nn.Parameter(weight_sum.view(-1, 1), requires_grad=False)
            
            # Sum the bias to a scalar
            self.bias_sum = nn.Parameter(torch.sum(linear.bias), requires_grad=False)
        
        # We don't need to keep the linear layer after precomputation
        del linear
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        # Ensure input is contiguous for optimal CUDA performance
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Use torch.addmm for efficient combined matrix multiplication and addition
        # This performs: bias_sum + x @ weight_sum in a single CUDA kernel call
        # beta=1.0: scale for bias_sum, alpha=1.0: scale for the matrix multiplication
        return torch.addmm(self.bias_sum, x, self.weight_sum)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 10
out_features = 5

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_features, out_features]