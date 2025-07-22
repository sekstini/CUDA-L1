import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    An optimized model that computes Kullback-Leibler Divergence for comparing two distributions.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        """
        Optimized KL divergence computation using direct mathematical formulation
        
        Args:
            predictions (torch.Tensor): Predicted probability distribution
            targets (torch.Tensor): Target probability distribution
            
        Returns:
            torch.Tensor: KL divergence loss (scalar)
        """
        # Direct KL computation: KL(P||Q) = sum(P * log(P/Q))
        # Using torch.xlogy for stability and efficiency
        # torch.xlogy(x, y) computes x * log(y) and handles x=0 case correctly
        
        # Ensure tensors are contiguous for better memory access patterns
        predictions_c = predictions if predictions.is_contiguous() else predictions.contiguous()
        targets_c = targets if targets.is_contiguous() else targets.contiguous()
        
        # Compute ratio P/Q directly
        ratio = targets_c / predictions_c
        
        # Use xlogy for efficient computation of P*log(P/Q)
        kl_terms = torch.xlogy(targets_c, ratio)
        
        # Sum over features, then mean over batch (equivalent to 'batchmean' reduction)
        return kl_terms.sum(dim=1).mean()

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape).softmax(dim=-1), torch.randn(batch_size, *input_shape).softmax(dim=-1)]

def get_init_inputs():
    return []