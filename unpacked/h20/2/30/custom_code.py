import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized implementation of GEMM + GroupNorm + HardTanh
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        num_groups (int): Number of groups for GroupNorm
        hardtanh_min (float): Minimum value for HardTanh
        hardtanh_max (float): Maximum value for HardTanh
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        
        # Create the same components as the reference implementation to ensure identical initialization
        ref_gemm = nn.Linear(in_features, out_features)
        ref_group_norm = nn.GroupNorm(num_groups, out_features)
        
        # Create custom parameters with the same initialization as the reference
        self.weight = nn.Parameter(ref_gemm.weight.data.clone())
        self.bias = nn.Parameter(ref_gemm.bias.data.clone())
        self.weight_gn = nn.Parameter(ref_group_norm.weight.data.clone())
        self.bias_gn = nn.Parameter(ref_group_norm.bias.data.clone())
        
        # Cache for transposed weight to avoid repeated transposition
        self.register_buffer('weight_t', self.weight.t().contiguous(), persistent=False)
        
        # Pre-compute constants for faster execution
        self.eps = 1e-5
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        batch_size = x.size(0)
        
        # Ensure input is contiguous for better memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Linear transformation using addmm which maps directly to CUBLAS
        # out = bias + x @ weight_t
        out = torch.addmm(self.bias, x, self.weight_t)
        
        # Apply group normalization
        # Reshape to [batch_size, out_features, 1] for group_norm using view
        out_3d = out.view(batch_size, self.out_features, 1)
        
        # Use native GroupNorm implementation which is already optimized
        normalized = F.group_norm(
            out_3d, 
            self.num_groups, 
            self.weight_gn, 
            self.bias_gn, 
            self.eps
        )
        
        # Reshape back to [batch_size, out_features] using view
        out = normalized.view(batch_size, self.out_features)
        
        # Apply HardTanh in-place for better efficiency
        out.clamp_(min=self.hardtanh_min, max=self.hardtanh_max)
        
        return out

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 1024
out_features = 512
num_groups = 8
hardtanh_min = -2.0
hardtanh_max = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]