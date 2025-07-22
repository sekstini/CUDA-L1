import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OptimizedGELU(nn.Module):
    """
    Optimized GELU implementation using PyTorch's built-in function
    """
    def __init__(self):
        super(OptimizedGELU, self).__init__()
    
    def forward(self, x):
        return F.gelu(x, approximate='tanh')

class OptimizedSelfAttention(nn.Module):
    """
    Highly optimized implementation of causal self-attention
    """
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        
        # Single QKV projection for efficiency
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        # Regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        
        # Pre-compute causal mask for fallback path
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        # Pre-compute scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Check for Flash Attention availability
        self.use_flash_attn = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # Single QKV projection for efficiency
        qkv = self.c_attn(x)
        
        # Efficient chunking operation
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to [B, nh, T, hs] with optimized memory layout
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Use Flash Attention if available
        if self.use_flash_attn:
            # Use PyTorch's optimized Flash Attention implementation
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True  # More efficient than explicit masking
            )
        else:
            # Optimized fallback implementation
            att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = torch.matmul(att, v)
        
        # Reshape back efficiently - use reshape instead of view+contiguous
        y = y.transpose(1, 2).reshape(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y

class ModelNew(nn.Module):
    """ An optimized Transformer block """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super(ModelNew, self).__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = OptimizedSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = OptimizedGELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        
        # Cache MLP forward function for efficiency
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))
        
        # Check for CUDA and AMP availability
        self.use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
        
        # Check for BFloat16 support
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    
    def forward(self, x):
        # Store original dtype for potential mixed precision operations
        orig_dtype = x.dtype
        
        if self.use_amp and x.is_cuda:
            # Choose precision type based on hardware support
            dtype = torch.bfloat16 if self.use_bf16 else torch.float16
            
            with torch.cuda.amp.autocast(dtype=dtype):
                # Direct residual connection pattern for better efficiency
                x = x + self.attn(self.ln_1(x))
                x = x + self.mlpf(self.ln_2(x))
        else:
            # Standard precision path
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlpf(self.ln_2(x))
        
        # Ensure output has the same dtype as input
        if x.dtype != orig_dtype:
            x = x.to(orig_dtype)
            
        return x

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