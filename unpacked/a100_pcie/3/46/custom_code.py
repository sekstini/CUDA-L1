import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * torch.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size
        
        # Pre-compute batch norm parameters for maximum efficiency
        self.register_buffer('bn_weight', None)
        self.register_buffer('bn_bias', None)
        self.register_buffer('bn_mean', None)
        self.register_buffer('bn_var_sqrt_inv', None)
        
    def _update_bn_params(self):
        """Pre-compute batch normalization parameters for efficient forward pass"""
        if (self.bn_weight is None or 
            self.bn_weight.device != self.clusters.device or
            not self.bn_weight.is_contiguous()):
            
            eps = self.batch_norm.eps
            self.bn_weight = self.batch_norm.weight.contiguous()
            self.bn_bias = self.batch_norm.bias.contiguous()
            self.bn_mean = self.batch_norm.running_mean.contiguous()
            self.bn_var_sqrt_inv = torch.rsqrt(self.batch_norm.running_var + eps).contiguous()

    def forward(self, x, mask=None):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (torch.Tensor): B x N x D

        Returns:
            (torch.Tensor): B x DK
        """
        batch_size, max_sample, _ = x.shape
        
        if x.device != self.clusters.device:
            msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)
        
        # Update batch norm parameters
        self._update_bn_params()
        
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Flatten input for matrix multiplication
        x_flat = x.view(-1, self.feature_size)  # BN x D
        
        # Ensure clusters are contiguous for optimal matrix multiplication
        clusters = self.clusters
        if not clusters.is_contiguous():
            clusters = clusters.contiguous()
        
        # Compute assignment using optimized matrix multiplication
        # BN x D @ D x (K+G) -> BN x (K+G)
        assignment = torch.mm(x_flat, clusters)
        
        # Apply batch normalization manually for efficiency
        # (x - mean) * var_sqrt_inv * weight + bias
        assignment = torch.addcmul(
            self.bn_bias,
            assignment.sub(self.bn_mean),
            self.bn_weight * self.bn_var_sqrt_inv
        )
        
        # Apply softmax and slice to remove ghost clusters
        assignment = F.softmax(assignment, dim=1)[:, :self.cluster_size]
        
        # Reshape assignment back to batch format
        # BN x K -> B x N x K
        assignment = assignment.view(batch_size, max_sample, self.cluster_size)
        
        # Transpose assignment for batch matrix multiplication
        # B x N x K -> B x K x N
        assignment_t = assignment.transpose(1, 2)
        
        # Optimize VLAD computation by transposing x once
        # B x N x D -> B x D x N
        x_t = x.transpose(1, 2)
        
        # Compute VLAD residuals using batch matrix multiplication
        # B x D x N @ B x N x K -> B x D x K
        vlad = torch.bmm(x_t, assignment)
        
        # Compute sum of assignments for each cluster
        # B x N x K -> B x 1 x K
        a_sum = torch.sum(assignment, dim=1, keepdim=True)
        
        # Compute weighted cluster centers
        # B x 1 x K * 1 x D x K -> B x D x K
        a = a_sum * self.clusters2
        
        # Subtract cluster centers (in-place to save memory)
        vlad.sub_(a)
        
        # L2 intra-normalization (normalize each feature dimension across clusters)
        vlad = F.normalize(vlad, p=2, dim=1)
        
        # Flatten and apply final L2 normalization
        # B x D x K -> B x DK
        vlad = vlad.view(batch_size, -1)
        
        # Final L2 normalization
        vlad = F.normalize(vlad, p=2, dim=1)
        
        return vlad

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 32
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 16

def get_inputs():
    return [torch.randn(batch_size, num_features, feature_size)]

def get_init_inputs():
    return [num_clusters, feature_size, ghost_clusters]