import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    An optimized model that computes Triplet Margin Loss for metric learning tasks.
    Uses optimized PyTorch operations for maximum performance.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.register_buffer('margin_tensor', torch.tensor([margin], dtype=torch.float32))
    
    def forward(self, anchor, positive, negative):
        # Ensure optimal memory layout for vectorized operations
        anchor = anchor.contiguous()
        positive = positive.contiguous()
        negative = negative.contiguous()
        
        # Compute differences directly to minimize intermediate allocations
        diff_pos = anchor - positive
        diff_neg = anchor - negative
        
        # Use torch.linalg.vector_norm which is highly optimized for GPU
        dist_pos = torch.linalg.vector_norm(diff_pos, ord=2, dim=1)
        dist_neg = torch.linalg.vector_norm(diff_neg, ord=2, dim=1)
        
        # Compute loss using vectorized operations
        # max(0, d_pos - d_neg + margin)
        losses = torch.relu(dist_pos - dist_neg + self.margin_tensor)
        
        # Use optimized mean reduction
        return losses.mean()

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [1.0]  # Default margin