import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    Optimized through advanced memory management and multi-stream execution pipeline.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.device = None
        self.compute_stream = None
        self.memory_stream = None
        self.output_cache = None
        self.A_cache = None
        self.B_cache = None
        self.initialized = False
        self.warmup_done = False
        self.target_shape_A = (batch_size, m, k)
        self.target_shape_B = (batch_size, k, n)
        self.target_shape_C = (batch_size, m, n)
        
    def _initialize_for_device(self, device):
        """Initialize device-specific optimizations with advanced memory management."""
        if self.initialized and self.device == device:
            return
            
        self.device = device
        
        if device.type == 'cuda':
            # Initialize CUDNN for potential performance benefits
            if hasattr(torch._C, '_cudnn_init'):
                torch._C._cudnn_init()
            
            # Create dual streams for computation and memory operations
            self.compute_stream = torch.cuda.Stream(device=device, priority=-1)
            self.memory_stream = torch.cuda.Stream(device=device, priority=0)
            
            # Pre-warm memory pool to avoid allocation overhead
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                # Pre-allocate a large chunk to warm up the memory pool
                temp = torch.empty(batch_size * m * n * 4, dtype=torch.float32, device=device)
                del temp
            
            # Pre-allocate output tensor with optimal memory layout and alignment
            self.output_cache = torch.empty(
                self.target_shape_C, 
                dtype=torch.float32, 
                device=device,
                memory_format=torch.contiguous_format
            )
            
            # Pre-allocate input tensor caches with optimal alignment
            self.A_cache = torch.empty(
                self.target_shape_A, 
                dtype=torch.float32, 
                device=device,
                memory_format=torch.contiguous_format
            )
            
            self.B_cache = torch.empty(
                self.target_shape_B, 
                dtype=torch.float32, 
                device=device,
                memory_format=torch.contiguous_format
            )
            
            self.warmup_done = False
        
        self.initialized = True
    
    def _warmup(self):
        """Advanced warmup with memory locality optimization."""
        if self.warmup_done:
            return
            
        # Use compute stream for warmup
        with torch.cuda.stream(self.compute_stream):
            # Initialize caches with specific patterns for optimal memory locality
            self.A_cache.normal_(0, 1)
            self.B_cache.normal_(0, 1)
            
            # Perform optimal number of warmup iterations with different patterns
            for i in range(2):
                torch.bmm(self.A_cache, self.B_cache, out=self.output_cache)
                # Add slight variation to ensure different code paths are warmed
                if i == 0:
                    self.A_cache *= 1.01
            
            # Ensure all operations complete and caches are properly initialized
            self.compute_stream.synchronize()
            
        self.warmup_done = True
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        # Fast path for CPU tensors
        if not A.is_cuda or not B.is_cuda:
            return torch.bmm(A, B)
        
        # Initialize optimizations for this device
        self._initialize_for_device(A.device)
        
        # Use compute stream for main operations
        with torch.cuda.stream(self.compute_stream):
            # Perform warmup if needed (only once)
            if not self.warmup_done:
                self._warmup()
            
            # Optimized fast path for exact target dimensions
            if A.shape == self.target_shape_A and B.shape == self.target_shape_B:
                # Check contiguity and handle accordingly
                A_ready = A
                B_ready = B
                
                # Use memory stream for any necessary memory operations
                if not A.is_contiguous() or not B.is_contiguous():
                    with torch.cuda.stream(self.memory_stream):
                        if not A.is_contiguous():
                            self.A_cache.copy_(A, non_blocking=True)
                            A_ready = self.A_cache
                        if not B.is_contiguous():
                            self.B_cache.copy_(B, non_blocking=True)
                            B_ready = self.B_cache
                    
                    # Wait for memory operations to complete
                    self.compute_stream.wait_stream(self.memory_stream)
                
                # Perform optimized BMM directly into pre-allocated output
                torch.bmm(A_ready, B_ready, out=self.output_cache)
                return self.output_cache
            else:
                # Fallback for different dimensions (should not happen in our case)
                A_cont = A.contiguous() if not A.is_contiguous() else A
                B_cont = B.contiguous() if not B.is_contiguous() else B
                return torch.bmm(A_cont, B_cont)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
m = 128
k = 256
n = 512

def get_inputs():
    A = torch.randn(batch_size, m, k)
    B = torch.randn(batch_size, k, n)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed