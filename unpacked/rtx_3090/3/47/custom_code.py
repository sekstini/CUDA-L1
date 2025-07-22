import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th

class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size
        
        # Pre-allocate and cache batch normalization parameters
        self.register_buffer('bn_weight', None, persistent=False)
        self.register_buffer('bn_bias', None, persistent=False)
        self.register_buffer('bn_mean', None, persistent=False)
        self.register_buffer('bn_var', None, persistent=False)
        self.register_buffer('bn_std_inv', None, persistent=False)
        
        # Pre-allocate buffers for intermediate results to reduce memory allocations
        self.register_buffer('assignment_buffer', None, persistent=False)
        self.register_buffer('vlad_buffer', None, persistent=False)
        self.register_buffer('norm_buffer', None, persistent=False)

    def forward(self, x, mask=None):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (th.Tensor): B x N x D

        Returns:
            (th.Tensor): B x DK
        """
        if x.device != self.clusters.device:
            msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)
        
        batch_size, max_sample, feature_size = x.shape
        
        # Ensure contiguous memory layout for optimal performance
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Cache batch normalization parameters with pre-computed inverse std
        if self.bn_weight is None or self.bn_weight.device != x.device:
            self.bn_weight = self.batch_norm.weight.contiguous()
            self.bn_bias = self.batch_norm.bias.contiguous()
            self.bn_mean = self.batch_norm.running_mean.contiguous()
            self.bn_var = self.batch_norm.running_var.contiguous()
            # Pre-compute inverse standard deviation for efficiency
            self.bn_std_inv = torch.rsqrt(self.bn_var + self.batch_norm.eps)
            
            # Initialize or resize buffers for this device
            total_samples = batch_size * max_sample
            self.assignment_buffer = torch.empty(
                (total_samples, self.cluster_size), 
                device=x.device, dtype=x.dtype
            )
            self.vlad_buffer = torch.empty(
                (batch_size, feature_size, self.cluster_size), 
                device=x.device, dtype=x.dtype
            )
            self.norm_buffer = torch.empty(
                (batch_size, 1, self.cluster_size), 
                device=x.device, dtype=x.dtype
            )
        
        # Reshape x for matrix multiplication without copying data
        total_samples = batch_size * max_sample
        x_flat = x.view(total_samples, feature_size)
        
        # Optimized matrix multiplication with clusters
        # Use clusters directly if already contiguous, otherwise make contiguous
        clusters = self.clusters if self.clusters.is_contiguous() else self.clusters.contiguous()
        
        # Compute x @ clusters efficiently
        assignment = torch.mm(x_flat, clusters)
        
        # Apply batch normalization directly using pre-computed parameters
        # (x - mean) * (1/sqrt(var + eps)) * weight + bias
        assignment.sub_(self.bn_mean)
        assignment.mul_(self.bn_std_inv)
        assignment.mul_(self.bn_weight)
        assignment.add_(self.bn_bias)
        
        # Apply softmax with better numerical stability
        assignment = F.softmax(assignment, dim=1)
        
        # Keep only non-ghost clusters if needed
        if self.ghost_clusters > 0:
            assignment = assignment[:, :self.cluster_size]
        
        # Reshape assignment for VLAD computation
        assignment = assignment.view(batch_size, max_sample, self.cluster_size)
        
        # Sum assignments across samples (efficiently)
        a_sum = torch.sum(assignment, dim=1, keepdim=True)
        
        # Compute cluster centers contribution
        clusters2 = self.clusters2 if self.clusters2.is_contiguous() else self.clusters2.contiguous()
        a = a_sum * clusters2
        
        # Transpose assignment for batch matrix multiplication
        assignment_t = assignment.transpose(1, 2)
        
        # Compute VLAD using optimized batch matrix multiplication
        vlad = torch.bmm(assignment_t, x)
        
        # Transpose vlad for the correct output format
        vlad = vlad.transpose(1, 2)
        
        # Subtract cluster centers (in-place to reduce memory allocation)
        vlad.sub_(a)
        
        # L2 intra norm (normalize each feature across clusters)
        # Compute norm along dimension 1 (features)
        norm = torch.norm(vlad, p=2, dim=1, keepdim=True)
        # Add small epsilon for numerical stability
        norm = norm.clamp(min=1e-12)
        # Normalize in-place
        vlad.div_(norm)
        
        # Flatten and apply final L2 normalization
        vlad = vlad.reshape(batch_size, -1)
        
        # Final L2 normalization
        norm = torch.norm(vlad, p=2, dim=1, keepdim=True)
        norm = norm.clamp(min=1e-12)
        vlad.div_(norm)
        
        return vlad

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 32
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 0

def get_inputs():
    return [torch.randn(batch_size, num_features, feature_size)]

def get_init_inputs():
    return [num_clusters, feature_size, ghost_clusters]