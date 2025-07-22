import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    An optimized model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Create a zeros tensor directly with the right shape
        # This avoids the select and unsqueeze operations in the reference implementation
        shape = list(x.shape)
        shape[self.dim] = 1
        zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
        
        # Concatenate zeros with x along self.dim and remove the last element
        # This creates a shifted version of x with a zero at the beginning
        exclusive_cumsum = torch.cat((zeros, x), dim=self.dim)[:-1]
        
        # Ensure the tensor is contiguous for optimal performance
        if not exclusive_cumsum.is_contiguous():
            exclusive_cumsum = exclusive_cumsum.contiguous()
        
        # Compute cumulative sum along self.dim
        return torch.cumsum(exclusive_cumsum, dim=self.dim)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]