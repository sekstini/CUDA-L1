import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D transposed convolution, followed by a sum, 
    layer normalization, average pooling, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        
        # Configure PyTorch for optimal performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        
        # Create the ConvTranspose3d layer
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        
        # Initialize other parameters
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()
        
        # Enable automatic mixed precision for Tensor Core utilization
        self.use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        
        # Set workspace limit for cuDNN to allow more efficient algorithms
        if torch.cuda.is_available():
            torch.backends.cudnn.workspace_limit = 1024 * 1024 * 1024  # 1GB
            
            # Convert weights to channels_last_3d format for better performance
            self.to('cuda')
            if hasattr(self.conv_transpose, 'weight'):
                self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(memory_format=torch.channels_last_3d)
            
            # Pre-warm the model to avoid compilation overhead
            try:
                dummy_input = torch.zeros(2, in_channels, 4, 4, 4, device='cuda')
                dummy_input = dummy_input.to(memory_format=torch.channels_last_3d)
                with torch.no_grad():
                    _ = self.forward(dummy_input)
            except Exception:
                pass

    def forward(self, x):
        # Convert to channels_last_3d for better memory access patterns
        if x.device.type == 'cuda':
            x = x.to(memory_format=torch.channels_last_3d)
            
        # Ensure input is contiguous in memory
        if not x.is_contiguous(memory_format=torch.channels_last_3d):
            x = x.contiguous(memory_format=torch.channels_last_3d)
            
        # Use automatic mixed precision if available
        if self.use_amp and x.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                # ConvTranspose3d operation - most computationally intensive
                x = self.conv_transpose(x)
                
                # Add sum_weight
                x = x + self.sum_weight
                
                # Layer normalization
                x = self.norm(x)
        else:
            # Fallback without AMP
            x = self.conv_transpose(x)
            x = x + self.sum_weight
            x = self.norm(x)
        
        # Average pooling and GELU activation (outside AMP context for better performance)
        x = self.avg_pool(x)
        x = self.gelu(x)
        
        return x

# Apply torch.compile with optimal settings
try:
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 256
    ModelNew = torch.compile(ModelNew, fullgraph=True)
except Exception as e:
    print(f"Failed to apply torch.compile: {e}")

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]