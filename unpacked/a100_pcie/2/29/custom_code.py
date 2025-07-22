import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        # CUDA graph related attributes
        self.graph_captured = False
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None
        
        # Create scripted version of the forward pass
        try:
            @torch.jit.script
            def scripted_forward(x, weight, bias):
                x = F.linear(x, weight, bias)
                x = F.mish(x)
                x = F.mish(x)
                return x
            
            self.scripted_forward = scripted_forward
        except Exception:
            self.scripted_forward = None
    
    def _initialize(self, x):
        """Initialize optimizations"""
        # Move model to the same device as input
        if self.linear.weight.device != x.device:
            self.linear = self.linear.to(x.device)
        
        # Perform warmup runs to ensure kernels are compiled
        with torch.no_grad():
            for _ in range(10):  # More thorough warmup for stability
                if self.scripted_forward is not None:
                    self.scripted_forward(x, self.linear.weight, self.linear.bias)
                else:
                    out = self.linear(x)
                    out = F.mish(out)
                    out = F.mish(out)
        
        # Ensure all operations are complete
        if x.is_cuda:
            torch.cuda.synchronize()
    
    def _capture_cuda_graph(self, x):
        """Capture CUDA graph for faster execution"""
        try:
            # Create static input tensor with optimal memory layout
            self.static_input = torch.empty_like(x, memory_format=torch.contiguous_format)
            self.static_input.copy_(x)
            
            # Run once to get output shape and allocate output tensor
            with torch.no_grad():
                if self.scripted_forward is not None:
                    result = self.scripted_forward(
                        self.static_input, 
                        self.linear.weight, 
                        self.linear.bias
                    )
                else:
                    result = self.linear(self.static_input)
                    result = F.mish(result)
                    result = F.mish(result)
            
            # Allocate static output with optimal memory layout
            self.static_output = torch.empty_like(result, memory_format=torch.contiguous_format)
            
            # Capture the CUDA graph
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graph):
                if self.scripted_forward is not None:
                    result = self.scripted_forward(
                        self.static_input, 
                        self.linear.weight, 
                        self.linear.bias
                    )
                else:
                    result = self.linear(self.static_input)
                    result = F.mish(result)
                    result = F.mish(result)
                self.static_output.copy_(result)
            
            # Ensure graph is ready
            torch.cuda.synchronize()
            self.graph_captured = True
            return True
        except Exception:
            # Fallback if graph capture fails
            self.graph_captured = False
            return False
    
    def forward(self, x):
        """
        Ultra-optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Ultra-optimized fast path for CUDA graph replay
        # This is the most critical path for performance - keep it as lean as possible
        if self.graph_captured and x.is_cuda:
            self.static_input.copy_(x)
            self.cuda_graph.replay()
            return self.static_output
        
        # Ensure input is contiguous for optimal performance
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Initialize if needed
        if not self.graph_captured and x.is_cuda:
            self._initialize(x)
            
            # Try to capture graph
            if self._capture_cuda_graph(x):
                # If capture succeeded, use the graph immediately
                self.static_input.copy_(x)
                self.cuda_graph.replay()
                return self.static_output
        
        # Standard execution path
        with torch.no_grad():
            if self.scripted_forward is not None:
                return self.scripted_forward(x, self.linear.weight, self.linear.bias)
            else:
                x = self.linear(x)
                x = F.mish(x)
                x = F.mish(x)
                return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 10
out_features = 20

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_features, out_features]