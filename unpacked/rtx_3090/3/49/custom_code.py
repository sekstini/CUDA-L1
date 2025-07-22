import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Mamba Structured State Space model implementation for benchmarking.
        
        :param batch_size: Size of the batch
        :param seq_length: Length of the input sequence
        :param n_heads: Number of attention heads
        :param d_head: Dimension of each head
        :param d_state: Dimension of the state space
        :param block_len: Length of each block for chunked computation
        """
        super(ModelNew, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        # Initialize parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        
        # Precompute number of chunks for efficiency
        self.n_chunks = seq_length // block_len
    
    def forward(self, X, initial_states=None):
        """
        Optimized forward pass implementing the SSD operation.
        
        :param X: Input tensor of shape (batch, length, n_heads, d_head)
        :param initial_states: Optional initial states
        :return: Final state tensor
        """
        batch_size = self.batch_size
        n_heads = self.n_heads
        d_head = self.d_head
        d_state = self.d_state
        n_chunks = self.n_chunks
        block_len = self.block_len
        
        # Reshape tensors efficiently using view instead of rearrange
        X_blocks = X.view(batch_size, n_chunks, block_len, n_heads, d_head)
        A_blocks = self.A.view(batch_size, n_chunks, block_len, n_heads)
        B_blocks = self.B.view(batch_size, n_chunks, block_len, n_heads, d_state)
        
        # Rearrange A for efficient processing
        A_blocks = A_blocks.permute(0, 3, 1, 2)  # b h c l
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        # Compute intra-chunk states efficiently
        A_end = A_cumsum[:, :, :, -1:]  # Last element of each block
        decay_states = torch.exp(A_end - A_cumsum)  # More stable computation
        
        # Optimize B * decay_states computation with broadcasting
        decay_states_expanded = decay_states.permute(0, 2, 3, 1).unsqueeze(-1)  # b c l h 1
        B_weighted = B_blocks * decay_states_expanded  # b c l h n
        
        # Compute states efficiently
        X_blocks_t = X_blocks.permute(0, 1, 3, 2, 4)  # b c h l p
        B_weighted_t = B_weighted.permute(0, 1, 3, 4, 2)  # b c h n l
        
        # Batch matrix multiplication for better performance
        states_t = torch.matmul(B_weighted_t, X_blocks_t)  # b c h n p
        states = states_t.permute(0, 1, 2, 4, 3)  # b c h p n
        
        # Handle initial states efficiently
        if initial_states is None:
            initial_states = torch.zeros(
                batch_size, 1, n_heads, d_head, d_state,
                device=X.device, dtype=X.dtype
            )
        
        # Concatenate states efficiently
        all_states = torch.cat([initial_states, states], dim=1)
        
        # Compute inter-chunk decay efficiently
        A_block_ends = A_cumsum[:, :, :, -1]  # b h c
        A_padded = F.pad(A_block_ends, (1, 0))  # b h (c+1)
        
        # Compute decay for final state calculation
        A_padded_cumsum = torch.cumsum(A_padded, dim=-1)
        last_cumsum = A_padded_cumsum[:, :, -1].unsqueeze(-1)  # b h 1
        decay_factors = torch.exp(last_cumsum - A_padded_cumsum)  # b h (c+1)
        
        # Optimize final state computation with broadcasting
        decay_expanded = decay_factors.unsqueeze(-1).unsqueeze(-1)  # b h (c+1) 1 1
        all_states_t = all_states.permute(0, 2, 1, 3, 4)  # b h (c+1) p n
        
        # Multiply and sum along chunk dimension
        final_states = (decay_expanded * all_states_t).sum(dim=2)  # b h p n
        
        return final_states

# Test parameters
batch_size = 16
seq_length = 128
n_heads = 8
d_head = 64
d_state = 16
block_len = 64

def get_inputs():
    return [torch.randn(batch_size, seq_length, n_heads, d_head)]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]