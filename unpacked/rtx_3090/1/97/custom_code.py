import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    A model that computes Cosine Similarity Loss for comparing vectors.
    Optimized implementation that maintains identical functionality.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Pre-allocate constants as buffers to avoid creating them during forward pass
        self.register_buffer('epsilon', torch.tensor(1e-8, dtype=torch.float32))
        self.register_buffer('one', torch.tensor(1.0, dtype=torch.float32))

    def forward(self, predictions, targets):
        # Ensure contiguous memory layout for better performance
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        
        # Compute dot products efficiently using direct elementwise multiplication
        # Sum along dimension 1 (vector dimension)
        dot_products = torch.sum(predictions * targets, dim=1)
        
        # Compute L2 norms efficiently (avoid using .pow(2) which creates a new tensor)
        # Using separate computations for better memory access patterns
        pred_norm_sq = torch.sum(predictions * predictions, dim=1)
        targ_norm_sq = torch.sum(targets * targets, dim=1)
        
        # Compute product of norms and add epsilon for numerical stability
        # Adding epsilon inside the sqrt is more numerically stable
        norm_product = pred_norm_sq * targ_norm_sq
        
        # Using rsqrt is more efficient than sqrt followed by division
        # Adding epsilon inside to prevent division by zero
        inv_prod_norms = torch.rsqrt(norm_product + self.epsilon)
        
        # Compute cosine similarity directly
        cosine_sim = dot_products * inv_prod_norms
        
        # Compute loss: mean of (1 - cosine_similarity)
        # Using pre-allocated constant for better performance
        loss = torch.mean(self.one - cosine_sim)
        
        return loss

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return []