import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.
    Optimized with CUDA graphs and in-place operations.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Pre-allocate buffers
        self.buffer = None
        
        # CUDA graph related attributes
        self.graph = None
        self.static_inputs = None
        self.static_output = None
        
        # Execution strategy flags
        self.use_cuda_graph = False
        self.initialized = False
    
    def _initialize(self, predictions, targets):
        """Initialize resources for the current input shapes and devices"""
        # Create buffer for intermediate results
        self.buffer = torch.empty_like(predictions)
        
        # Only try to initialize CUDA graph if inputs are on CUDA
        if predictions.is_cuda:
            try:
                # Create static tensors for inputs and output
                self.static_inputs = [
                    predictions.clone(),
                    targets.clone()
                ]
                self.static_output = torch.empty([], device=predictions.device, dtype=torch.float32)
                
                # Capture the graph
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    # Compute 1 - predictions * targets directly
                    torch.mul(self.static_inputs[0], self.static_inputs[1], out=self.buffer)
                    self.buffer.neg_()
                    self.buffer.add_(1.0)
                    self.buffer.clamp_(min=0)
                    
                    # Compute mean efficiently
                    result = torch.mean(self.buffer)
                    self.static_output.copy_(result)
                
                self.use_cuda_graph = True
                
                # Single warmup replay for better performance
                self.graph.replay()
                    
            except Exception:
                # If graph capture fails, disable CUDA graph usage
                self.use_cuda_graph = False
                self.graph = None
                self.static_inputs = None
                self.static_output = None
        
        self.initialized = True
    
    def forward(self, predictions, targets):
        # Initialize if needed
        if not self.initialized:
            if not predictions.is_contiguous():
                predictions = predictions.contiguous()
            if not targets.is_contiguous():
                targets = targets.contiguous()
            self._initialize(predictions, targets)
            
        # Fast path using CUDA graphs if available
        if self.use_cuda_graph:
            # Copy inputs to static tensors - no checks needed in hot path
            self.static_inputs[0].copy_(predictions)
            self.static_inputs[1].copy_(targets)
            
            # Replay the graph
            self.graph.replay()
            
            # Return the result
            return self.static_output
        
        # Fallback to optimized PyTorch implementation
        if not predictions.is_contiguous():
            predictions = predictions.contiguous()
        if not targets.is_contiguous():
            targets = targets.contiguous()
            
        if self.buffer is None or self.buffer.shape != predictions.shape or self.buffer.device != predictions.device:
            self.buffer = torch.empty_like(predictions)
        
        # Compute 1 - predictions * targets directly with in-place operations
        torch.mul(predictions, targets, out=self.buffer)
        self.buffer.neg_()
        self.buffer.add_(1.0)
        self.buffer.clamp_(min=0)
        return torch.mean(self.buffer)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_shape = (1,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randint(0, 2, (batch_size, 1)).float() * 2 - 1]

def get_init_inputs():
    return []