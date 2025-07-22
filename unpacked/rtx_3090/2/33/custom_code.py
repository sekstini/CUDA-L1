import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized implementation that fuses GEMM, scaling, and batch normalization
    operations for improved performance.
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        scale_shape (tuple): Shape of the scaling factor
        eps (float): Small constant added to the denominator for numerical stability
        momentum (float): Momentum factor for running statistics
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.momentum = momentum
        
        # Create standard components as in the reference implementation
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        
        # Pre-computed buffers for optimized inference
        self.register_buffer('fused_weight', torch.zeros_like(self.gemm.weight))
        self.register_buffer('fused_bias', torch.zeros_like(self.gemm.bias))
        
        # Ultra-efficient parameter tracking system
        self.register_buffer('weights_version', torch.tensor([0], dtype=torch.long))
        self.register_buffer('bn_stats_version', torch.tensor([0], dtype=torch.long))
        self.register_buffer('was_training', torch.tensor([True], dtype=torch.bool))
        
        # Pre-reshape buffers for efficient broadcasting
        self.register_buffer('gamma_div_std', torch.zeros(out_features))
        self.register_buffer('scale_reshaped', torch.zeros(out_features, 1))
        
        # Initialize fused parameters
        self._update_fused_parameters(force=True)
    
    def _check_update_needed(self):
        """Ultra-efficient hierarchical check if parameters need updating"""
        # Tier 1: Fast path - check training mode change (most common scenario)
        training_changed = self.training != self.was_training.item()
        if training_changed:
            self.was_training.fill_(self.training)
            return True
            
        # If in training mode, no need to check further
        if self.training:
            return False
            
        # Tier 2: Check if BN statistics have changed (common during inference)
        current_bn_stats_version = (
            self.bn.running_mean._version + 
            self.bn.running_var._version
        )
        
        stats_changed = current_bn_stats_version != self.bn_stats_version.item()
        if stats_changed:
            self.bn_stats_version.fill_(current_bn_stats_version)
            
            # Tier 3: Check if weights have changed (less frequent)
            current_weights_version = (
                self.gemm.weight._version + 
                self.gemm.bias._version + 
                self.scale._version + 
                self.bn.weight._version + 
                self.bn.bias._version
            )
            
            weights_changed = current_weights_version != self.weights_version.item()
            if weights_changed:
                self.weights_version.fill_(current_weights_version)
                
            return True
            
        return False
    
    def _update_fused_parameters(self, force=False):
        """Update pre-computed parameters for fused inference"""
        if self.training and not force:
            return
            
        with torch.no_grad():
            # Pre-compute standard deviation with eps for numerical stability
            std = torch.sqrt(self.bn.running_var + self.eps)
            
            # Pre-compute gamma/std for efficiency
            self.gamma_div_std.copy_(self.bn.weight / std)
            
            # Update pre-reshaped scale buffer
            self.scale_reshaped.copy_(self.scale.view(-1, 1))
            
            # Fused weight: gamma * W * scale / sqrt(var + eps)
            # Use optimized memory access pattern with pre-reshaped tensors
            gamma_div_std_reshaped = self.gamma_div_std.view(-1, 1)
            self.fused_weight.copy_(
                self.gemm.weight * (gamma_div_std_reshaped * self.scale_reshaped)
            )
            
            # Fused bias: gamma * bias * scale / sqrt(var + eps) - gamma * mean / sqrt(var + eps) + beta
            # Compute in stages for better numerical stability and memory access
            scale_flat = self.scale.view(-1)
            self.fused_bias.copy_(
                self.gemm.bias * self.gamma_div_std * scale_flat - 
                self.bn.running_mean * self.gamma_div_std + 
                self.bn.bias
            )
    
    def forward(self, x):
        if self.training:
            # Training path: use functional interfaces for better performance
            # while maintaining correct gradient computation
            
            # Step 1: Linear transformation (GEMM)
            # Use F.linear directly instead of self.gemm for better performance
            linear_out = F.linear(x, self.gemm.weight, self.gemm.bias)
            
            # Step 2: Apply scaling
            # Use in-place multiplication if possible
            if linear_out.requires_grad:
                scaled_out = linear_out * self.scale
            else:
                scaled_out = linear_out.mul_(self.scale)
            
            # Step 3: Apply batch normalization
            return F.batch_norm(
                scaled_out,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                self.training,
                self.bn.momentum,
                self.eps
            )
        else:
            # Inference path: use fused implementation
            if self._check_update_needed():
                self._update_fused_parameters()
            
            # Single fused operation
            return F.linear(x, self.fused_weight, self.fused_bias)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 1024
out_features = 512
scale_shape = (out_features,)

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_features, out_features, scale_shape]