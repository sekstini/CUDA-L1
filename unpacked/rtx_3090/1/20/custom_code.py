import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Simple model that performs a LeakyReLU activation with ultra-minimal overhead optimization.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LeakyReLU activation to the input tensor with absolute minimal overhead.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        # Check if we can use in-place operation for maximum efficiency
        if x.requires_grad:
            # If gradients are required, we cannot use in-place operations
            # Use the most direct path to PyTorch's C++ backend
            return torch._C._nn.leaky_relu(x, self.negative_slope)
        else:
            # For inference, try in-place operation for maximum memory efficiency
            # This eliminates memory allocation overhead completely
            try:
                return torch._C._nn.leaky_relu_(x, self.negative_slope)
            except:
                # Fallback to non-in-place if in-place fails
                return torch._C._nn.leaky_relu(x, self.negative_slope)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed