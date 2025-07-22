import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features  
        add_value_shape (tuple): Shape of the add_value parameter
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        # Use nn.Linear to ensure identical weight/bias initialization
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        
        # Extract weights and bias for direct access
        self.weight = self.matmul.weight
        self.bias = self.matmul.bias
        
        # Pre-compute weight transpose for efficiency
        self.register_buffer('weight_t', self.weight.t().contiguous())
        
        # Pre-compute combined bias for efficiency
        self.register_buffer('combined_bias', self.bias + self.add_value)
        
        # Pre-allocate buffers for intermediate results
        self.buffer1 = None
        self.buffer2 = None
        
        # Cache parameter versions using a lightweight tracking mechanism
        self._weight_version = self.weight._version
        self._bias_version = self.bias._version
        self._add_value_version = self.add_value._version
    
    def forward(self, x):
        """
        Optimized forward pass with minimal overhead
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Initialize or resize buffers if needed
        batch_size = x.size(0)
        if self.buffer1 is None or self.buffer1.size(0) != batch_size:
            self.buffer1 = torch.empty(batch_size, self.weight.size(0), 
                                      dtype=x.dtype, device=x.device)
            self.buffer2 = torch.empty_like(self.buffer1)
        
        # Update cached parameters if needed using direct version checks
        # This approach is more efficient than tuple comparison
        if (self.weight._version != self._weight_version or 
            self.bias._version != self._bias_version or 
            self.add_value._version != self._add_value_version):
            with torch.no_grad():
                self.weight_t.copy_(self.weight.t().contiguous())
                self.combined_bias.copy_(self.bias + self.add_value)
            self._weight_version = self.weight._version
            self._bias_version = self.bias._version
            self._add_value_version = self.add_value._version
        
        # Step 1: Fused matrix multiplication and bias addition
        # Using addmm for fused matrix multiplication and bias addition
        torch.addmm(self.combined_bias, x, self.weight_t, beta=1.0, alpha=1.0, out=self.buffer1)
        
        # Step 2: Apply Swish activation (x * sigmoid(x))
        # Using buffer2 for sigmoid result
        torch.sigmoid(self.buffer1, out=self.buffer2)
        # Multiply in-place with original values
        self.buffer2.mul_(self.buffer1)
        
        # Step 3: Apply Tanh activation in-place
        self.buffer2.tanh_()
        
        # Step 4: Apply GELU activation
        # Using buffer1 as output to alternate buffer usage
        torch.nn.functional.gelu(self.buffer2, out=self.buffer1)
        
        # Step 5: Apply Hardtanh activation in-place (equivalent to clamp)
        self.buffer1.clamp_(-1.0, 1.0)
        
        return self.buffer1

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 1024
out_features = 512
add_value_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]