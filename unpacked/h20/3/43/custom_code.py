import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelNew(nn.Module):
    """
    An optimized implementation of MinGPT's CausalSelfAttention using Flash Attention
    """
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        # Pre-compute scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Check if Flash Attention is available (PyTorch >= 2.0)
        self.has_flash_attn = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch
        # Compute QKV projections in a single operation for efficiency
        qkv = self.c_attn(x)  # (B, T, 3*C)
        
        if self.has_flash_attn:
            # Optimized path using Flash Attention
            # Split QKV and reshape efficiently
            q, k, v = qkv.chunk(3, dim=2)
            
            # Reshape: [B, T, C] -> [B, T, nh, C/nh] -> [B, nh, T, C/nh]
            # Use view instead of reshape when possible for better performance
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
            
            # Use PyTorch's optimized Flash Attention implementation
            # This avoids materializing the full attention matrix
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,  # Not needed when is_causal=True
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,  # This handles causal masking efficiently
                scale=self.scale
            )
        else:
            # Fallback to standard implementation for older PyTorch versions
            # Split QKV and reshape
            q, k, v = qkv.chunk(3, dim=2)
            
            # Reshape: [B, T, C] -> [B, T, nh, C/nh] -> [B, nh, T, C/nh]
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

            # Causal self-attention
            att = (q @ k.transpose(-2, -1)) * self.scale
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        # Reshape: [B, nh, T, head_dim] -> [B, T, C]
        # Use contiguous to ensure optimal memory layout
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.randn(batch_size, seq_len, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]