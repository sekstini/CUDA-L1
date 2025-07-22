import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A * B) for upper triangular matrices.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.streams = None
        self.stream_initialized = False
    
    def _initialize_streams(self, device):
        """Initialize CUDA streams for optimal parallelization."""
        if device.type == 'cuda' and not self.stream_initialized:
            try:
                # Use 8 streams for optimal parallelization based on previous performance analysis
                num_streams = 8
                self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
                self.stream_initialized = True
            except:
                self.streams = None
                self.stream_initialized = True
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        N = A.shape[0]
        device = A.device
        dtype = A.dtype
        
        # For small matrices, use reference implementation to avoid block overhead
        if N < 512:
            return torch.triu(torch.matmul(A, B))
        
        # Initialize streams if needed
        self._initialize_streams(device)
        
        # Pre-allocate output tensor with zeros for optimal memory usage
        C = torch.zeros((N, N), device=device, dtype=dtype)
        
        # Optimized block size for N=4096
        block_size = 512
        
        # Phase 1: Process diagonal blocks first for maximum data reuse
        for i in range(0, N, block_size):
            i_end = min(i + block_size, N)
            j = i  # Diagonal block
            j_end = min(j + block_size, N)
            
            # Process diagonal block
            self._process_block(A, B, C, i, i_end, j, j_end, is_diagonal=True)
        
        # Phase 2: Process off-diagonal blocks with optimized stream distribution
        if device.type == 'cuda' and self.streams:
            # Collect all off-diagonal blocks with computational complexity metrics
            off_diagonal_blocks = []
            for j in range(block_size, N, block_size):  # Column-major order for better locality
                j_end = min(j + block_size, N)
                for i in range(0, j, block_size):  # Only upper triangular blocks
                    i_end = min(i + block_size, N)
                    
                    # Calculate computational complexity for load balancing
                    k_start = i
                    k_end = min(j_end, N)
                    
                    # Skip blocks that would be zero
                    if k_start >= k_end:
                        continue
                    
                    # Complexity is proportional to output block size * k-range
                    complexity = (i_end - i) * (j_end - j) * (k_end - k_start)
                    off_diagonal_blocks.append((i, i_end, j, j_end, complexity))
            
            # Sort blocks by computational complexity for optimal load balancing
            off_diagonal_blocks.sort(key=lambda x: x[4], reverse=True)
            
            # Distribute blocks across streams with advanced load balancing
            stream_workloads = [0] * len(self.streams)
            block_to_stream = {}
            
            # Assign blocks to streams using a greedy load balancing approach
            for i, i_end, j, j_end, complexity in off_diagonal_blocks:
                # Find the stream with the least workload
                min_load_idx = stream_workloads.index(min(stream_workloads))
                block_to_stream[(i, i_end, j, j_end)] = min_load_idx
                stream_workloads[min_load_idx] += complexity
            
            # Process blocks using the assigned streams
            active_streams = set()
            for (i, i_end, j, j_end), stream_idx in block_to_stream.items():
                stream = self.streams[stream_idx]
                active_streams.add(stream)
                
                # Process block asynchronously
                with torch.cuda.stream(stream):
                    self._process_block(A, B, C, i, i_end, j, j_end, is_diagonal=False)
            
            # Synchronize only active streams for efficiency
            for stream in active_streams:
                stream.synchronize()
        else:
            # CPU fallback: process off-diagonal blocks sequentially in column-major order
            for j in range(block_size, N, block_size):
                j_end = min(j + block_size, N)
                for i in range(0, j, block_size):  # Only upper triangular blocks
                    i_end = min(i + block_size, N)
                    self._process_block(A, B, C, i, i_end, j, j_end, is_diagonal=False)
        
        return C
    
    def _process_block(self, A, B, C, i, i_end, j, j_end, is_diagonal):
        """
        Process a single block of the matrix multiplication with mathematical optimization.
        
        Args:
            A (torch.Tensor): First input matrix
            B (torch.Tensor): Second input matrix  
            C (torch.Tensor): Output matrix
            i, i_end: Row range for the current block
            j, j_end: Column range for the current block
            is_diagonal: Whether this is a diagonal block
        """
        # Mathematical optimization: precise k-range based on upper triangular properties
        # A[i,k] is zero for k < i, B[k,j] is zero for k > j
        k_start = i
        k_end = min(j_end, A.shape[1])  # Ensure we don't exceed matrix bounds
        
        # Early termination for zero blocks
        if k_start >= k_end:
            return
        
        # Extract relevant blocks with optimal memory access
        A_block = A[i:i_end, k_start:k_end]
        B_block = B[k_start:k_end, j:j_end]
        C_slice = C[i:i_end, j:j_end]
        
        # Use optimized BLAS operation with in-place output
        torch.addmm(C_slice, A_block, B_block, beta=0, alpha=1, out=C_slice)
        
        # Apply upper triangular mask only to diagonal blocks
        if is_diagonal:
            C_slice.triu_()

N = 4096

def get_inputs():
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: A list containing two upper triangular matrices of shape (N, N).
    """
    A = torch.triu(torch.randn(N, N))
    B = torch.triu(torch.randn(N, N))
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs are needed for this model.

    Returns:
        list: An empty list.
    """
    return []