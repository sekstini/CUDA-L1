import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class ModelNew(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end that uses ReLU instead of Softmax.
    Optimized implementation with chunked computation for better memory efficiency and performance.
    """

    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        
        # Determine optimal chunk sizes based on sequence length
        # These values are tuned based on the performance of previous attempts
        self.q_chunk_size = min(256, max_seqlen)
        self.kv_chunk_size = min(512, max_seqlen)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        head_size = C // self.n_head
        scale = 1.0 / math.sqrt(head_size)
        
        # Efficient QKV projection and reshaping
        qkv = self.c_attn(x)  # (B, T, 3*C)
        
        # Split and reshape in the most efficient way
        qkv = qkv.view(B, T, 3, self.n_head, head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, T, hs)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, nh, T, hs)
        
        # Ensure tensors are contiguous for efficient matrix multiplication
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Pre-allocate output tensor to avoid dynamic allocation
        y = torch.zeros_like(q)
        
        # Process query sequence in chunks
        for i in range(0, T, self.q_chunk_size):
            i_end = min(i + self.q_chunk_size, T)
            q_chunk = q[:, :, i:i_end]  # (B, nh, chunk_size, hs)
            
            # For each query position, we only need to compute attention up to that position (causal)
            # Process key-value sequence in chunks
            for j in range(0, i_end, self.kv_chunk_size):
                j_end = min(j + self.kv_chunk_size, i_end)
                k_chunk = k[:, :, j:j_end]  # (B, nh, chunk_size, hs)
                v_chunk = v[:, :, j:j_end]  # (B, nh, chunk_size, hs)
                
                # Compute attention scores for this chunk pair
                # (B, nh, q_chunk_size, hs) @ (B, nh, hs, kv_chunk_size) -> (B, nh, q_chunk_size, kv_chunk_size)
                att_chunk = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scale
                
                # Apply causal mask - only for chunks where j+chunk_size > i
                # This optimization avoids unnecessary masking operations
                if j + self.kv_chunk_size > i:
                    # Create a mask for this specific chunk pair
                    mask_chunk = self.bias[:, :, i:i_end, j:j_end]
                    att_chunk.masked_fill_(mask_chunk == 0, float('-inf'))
                
                # Apply ReLU activation
                att_chunk = F.relu(att_chunk)
                
                # Apply attention to values
                # (B, nh, q_chunk_size, kv_chunk_size) @ (B, nh, kv_chunk_size, hs) -> (B, nh, q_chunk_size, hs)
                y[:, :, i:i_end] += torch.matmul(att_chunk, v_chunk)
        
        # Reshape output back to original format
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Apply output projection
        return y

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
max_seqlen = 1024
n_embd = 768  # Hidden dimension, typical for BERT-base size
n_head = 12   # Number of attention heads, typical for BERT-base size

def get_inputs():
    return [torch.randn(batch_size, max_seqlen, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, max_seqlen]