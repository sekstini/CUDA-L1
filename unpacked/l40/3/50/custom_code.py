import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalReLUAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, causal_mask=None):
        # q, k, v: [B, nh, T, hs]
        B, nh, T, hs = q.size()
        
        # Ensure contiguous memory layout for optimal access
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Initialize output tensor
        output = torch.zeros_like(v)
        
        # Determine optimal block size based on sequence length
        # For longer sequences, use smaller blocks to reduce memory pressure
        block_size = min(256, T // 2) if T > 512 else min(256, T)
        
        # Process attention in blocks to avoid materializing the full attention matrix
        for i in range(0, T, block_size):
            i_end = min(i + block_size, T)
            q_block = q[:, :, i:i_end, :]  # [B, nh, block_size, hs]
            
            # For causal attention, we only need to consider keys up to the current query position
            for j in range(0, i_end, block_size):
                j_end = min(j + block_size, i_end)
                k_block = k[:, :, j:j_end, :]  # [B, nh, block_size, hs]
                v_block = v[:, :, j:j_end, :]  # [B, nh, block_size, hs]
                
                # Compute attention scores for this block
                att_scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale  # [B, nh, q_block, k_block]
                
                # Apply causal masking - only needed at the boundary block
                if j <= i < j_end:
                    # Create causal mask for this block
                    q_indices = torch.arange(i, i_end, device=q.device).unsqueeze(1)
                    k_indices = torch.arange(j, j_end, device=q.device).unsqueeze(0)
                    mask_block = q_indices >= k_indices
                    mask_block = mask_block.unsqueeze(0).unsqueeze(0)  # [1, 1, q_block, k_block]
                    
                    # Apply mask
                    att_scores.masked_fill_(~mask_block, float('-inf'))
                
                # Apply ReLU activation
                att_scores = F.relu(att_scores)
                
                # Apply attention to values and accumulate
                output[:, :, i:i_end, :] += torch.matmul(att_scores, v_block)
        
        # Save tensors needed for backward
        ctx.save_for_backward(q, k, v, causal_mask)
        ctx.scale = scale
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, causal_mask = ctx.saved_tensors
        scale = ctx.scale
        B, nh, T, hs = q.size()
        
        # Initialize gradients
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        
        # Determine optimal block size
        block_size = min(256, T // 2) if T > 512 else min(256, T)
        
        # Process in blocks for memory efficiency
        for i in range(0, T, block_size):
            i_end = min(i + block_size, T)
            q_block = q[:, :, i:i_end, :]
            grad_output_block = grad_output[:, :, i:i_end, :]
            
            for j in range(0, i_end, block_size):
                j_end = min(j + block_size, i_end)
                k_block = k[:, :, j:j_end, :]
                v_block = v[:, :, j:j_end, :]
                
                # Compute attention scores
                att_scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale
                
                # Apply causal masking if needed
                if j <= i < j_end:
                    # Create causal mask for this block
                    q_indices = torch.arange(i, i_end, device=q.device).unsqueeze(1)
                    k_indices = torch.arange(j, j_end, device=q.device).unsqueeze(0)
                    mask_block = q_indices >= k_indices
                    mask_block = mask_block.unsqueeze(0).unsqueeze(0)  # [1, 1, q_block, k_block]
                    
                    # Apply mask
                    att_scores.masked_fill_(~mask_block, float('-inf'))
                
                # Apply ReLU and store for gradient computation
                att_relu = F.relu(att_scores)
                
                # Gradient for v
                grad_v[:, :, j:j_end, :] += torch.matmul(att_relu.transpose(-2, -1), grad_output_block)
                
                # Gradient for attention scores
                grad_att = torch.matmul(grad_output_block, v_block.transpose(-2, -1))
                
                # Apply ReLU gradient (derivative is 1 where input > 0, 0 otherwise)
                grad_att = grad_att * (att_relu > 0).float()
                
                # Gradients for q and k
                grad_q[:, :, i:i_end, :] += torch.matmul(grad_att, k_block) * scale
                grad_k[:, :, j:j_end, :] += torch.matmul(grad_att.transpose(-2, -1), q_block) * scale
        
        return grad_q, grad_k, grad_v, None, None

class ModelNew(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end that uses ReLU instead of Softmax.
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
        
        # Precompute scale factor for efficiency
        self.scale = 1.0 / math.sqrt(n_embd // n_head)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=2)
        
        # Reshape and transpose in one step for better efficiency
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Use our optimized causal ReLU attention
        y = CausalReLUAttentionFunction.apply(q, k, v, self.scale, self.bias[:,:,:T,:T])
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

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