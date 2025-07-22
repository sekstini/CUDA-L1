import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized implementation of matrix-vector multiplication (C = A * B).
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.warmed_up = False
        self.stream = None
        self.output_cache = None
        self.traced_model = None
        self.compile_available = hasattr(torch, 'compile')
        self.compiled_mm = None
        self.optimization_level = 0
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        # Ensure inputs are contiguous for optimal memory access
        A = A.contiguous()
        B = B.contiguous()
        
        # Create dedicated CUDA stream for GPU operations if on GPU
        if self.stream is None and torch.cuda.is_available() and A.is_cuda:
            self.stream = torch.cuda.Stream()
        
        # Pre-allocate output tensor with optimal memory layout
        if self.output_cache is None or self.output_cache.shape != (A.shape[0], B.shape[1]) or self.output_cache.device != A.device:
            self.output_cache = torch.empty(
                (A.shape[0], B.shape[1]), 
                dtype=A.dtype, 
                device=A.device,
                memory_format=torch.contiguous_format
            )
        
        # Comprehensive warm-up and optimization on first call
        if not self.warmed_up and A.is_cuda:
            with torch.cuda.stream(self.stream) if self.stream is not None else torch.no_grad():
                # Extended warm-up for thorough JIT compilation
                for _ in range(20):
                    _ = torch.mm(A, B)
                
                # Create optimized models
                self._create_optimized_models(A, B)
                
                # Ensure all operations complete
                if self.stream is not None:
                    self.stream.synchronize()
                
                self.warmed_up = True
        
        # Execute with highest available optimization level
        execution_context = torch.cuda.stream(self.stream) if self.stream is not None and A.is_cuda else torch.no_grad()
        
        with execution_context:
            # Try compiled function if available (PyTorch 2.0+)
            if self.compiled_mm is not None and self.optimization_level >= 3:
                try:
                    result = self.compiled_mm(A, B)
                    self.output_cache.copy_(result)
                    return self.output_cache
                except Exception:
                    pass
            
            # Try traced model (with optimization)
            if self.traced_model is not None and self.optimization_level >= 1:
                try:
                    result = self.traced_model(A, B)
                    self.output_cache.copy_(result)
                    return self.output_cache
                except Exception:
                    pass
            
            # Fallback to optimized torch.mm with pre-allocated output
            torch.mm(A, B, out=self.output_cache)
            return self.output_cache
    
    def _create_optimized_models(self, A, B):
        """Create multiple levels of optimized models."""
        def mm_func(a, b):
            return torch.mm(a, b)
        
        try:
            # Level 1: JIT Trace with optimization
            self.traced_model = torch.jit.trace(mm_func, (A, B))
            self.traced_model = torch.jit.optimize_for_inference(self.traced_model)
            self.optimization_level = max(self.optimization_level, 1)
            
            # Additional warm-up for traced model
            for _ in range(10):
                _ = self.traced_model(A, B)
            
            # Level 3: Try torch.compile if available (PyTorch 2.0+)
            if self.compile_available:
                try:
                    self.compiled_mm = torch.compile(mm_func)
                    # Warm up compiled function
                    for _ in range(5):
                        _ = self.compiled_mm(A, B)
                    self.optimization_level = max(self.optimization_level, 3)
                except Exception:
                    pass
                
        except Exception:
            # Fallback to basic implementation
            self.traced_model = None
            self.optimization_level = 0

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
M = 256
K = 131072

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, 1)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed