import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Mamba Structured State Space model implementation for benchmarking.
        
        :param batch_size: Size of the batch
        :param seq_length: Length of the input sequence
        :param n_heads: Number of attention heads
        :param d_head: Dimension of each head
        :param d_state: Dimension of the state space
        :param block_len: Length of each block for chunked computation
        """
        super(ModelNew, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        # Initialize parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        
        # Try to load CUDA kernel if available
        self.cuda_kernel = None
        if torch.cuda.is_available():
            try:
                self.cuda_kernel = self._load_cuda_kernel()
            except Exception as e:
                print(f"Failed to load CUDA kernel: {e}")
                self.cuda_kernel = None
    
    def _load_cuda_kernel(self):
        """Load the CUDA kernel for optimized computation."""
        from torch.utils.cpp_extension import load_inline
        
        cuda_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        
        // Fast math operations
        #define FAST_EXP __expf
        
        template <typename scalar_t>
        __global__ void mamba_final_state_kernel(
            const scalar_t* __restrict__ X,        // [batch_size, seq_length, n_heads, d_head]
            const scalar_t* __restrict__ A,        // [batch_size, seq_length, n_heads]
            const scalar_t* __restrict__ B,        // [batch_size, seq_length, n_heads, d_state]
            scalar_t* __restrict__ output,         // [batch_size, n_heads, d_head, d_state]
            const scalar_t* __restrict__ initial_states, // [batch_size, n_heads, d_head, d_state] or nullptr
            const int batch_size,
            const int seq_length,
            const int n_heads,
            const int d_head,
            const int d_state,
            const int block_len) {
            
            // Calculate indices
            const int batch_idx = blockIdx.x;
            const int head_idx = blockIdx.y;
            const int d_state_idx = threadIdx.x;
            const int d_head_base_idx = threadIdx.y * 16; // Each thread processes 16 elements in d_head dimension
            
            // Early exit if indices are out of bounds
            if (batch_idx >= batch_size || head_idx >= n_heads || d_state_idx >= d_state)
                return;
            
            // Number of chunks
            const int n_chunks = seq_length / block_len;
            
            // Use shared memory for A values and cumulative sums
            extern __shared__ float shared_mem[];
            float* A_chunk = shared_mem;                      // Size: block_len + 32 (padding)
            float* A_cumsum = &A_chunk[block_len + 32];       // Size: n_chunks + 1 + 32 (padding)
            
            // Initialize A_cumsum with zeros
            if (threadIdx.y == 0 && d_state_idx < n_chunks + 1) {
                A_cumsum[d_state_idx] = 0.0f;
            }
            __syncthreads();
            
            // Compute chunk sums in parallel
            if (threadIdx.y == 0 && d_state_idx < n_chunks) {
                float chunk_sum = 0.0f;
                
                // Load and sum A values for this chunk
                for (int l = 0; l < block_len; l++) {
                    int seq_idx = d_state_idx * block_len + l;
                    float a_val = A[batch_idx * seq_length * n_heads + seq_idx * n_heads + head_idx];
                    chunk_sum += a_val;
                }
                
                // Store chunk sum for later cumulative sum calculation
                A_cumsum[d_state_idx + 1] = chunk_sum;
            }
            __syncthreads();
            
            // Compute cumulative sums sequentially
            if (threadIdx.y == 0 && d_state_idx == 0) {
                for (int c = 1; c <= n_chunks; c++) {
                    A_cumsum[c] += A_cumsum[c-1];
                }
            }
            __syncthreads();
            
            // Get the last cumulative sum value
            float last_cumsum = A_cumsum[n_chunks];
            
            // Process each element in the d_head dimension assigned to this thread
            for (int d_head_offset = 0; d_head_offset < 16 && d_head_base_idx + d_head_offset < d_head; d_head_offset++) {
                int d_head_idx = d_head_base_idx + d_head_offset;
                
                // Initialize final state
                float final_state = 0.0f;
                
                // Set initial state if provided
                if (initial_states != nullptr) {
                    final_state = initial_states[
                        batch_idx * n_heads * d_head * d_state + 
                        head_idx * d_head * d_state + 
                        d_head_idx * d_state + 
                        d_state_idx
                    ];
                    
                    // Apply decay to initial state
                    float decay_initial = FAST_EXP(last_cumsum - A_cumsum[0]);
                    final_state *= decay_initial;
                }
                
                // Process each chunk
                for (int c = 0; c < n_chunks; c++) {
                    // Load A values for this chunk into shared memory
                    if (threadIdx.y == 0 && d_state_idx < block_len) {
                        int seq_idx = c * block_len + d_state_idx;
                        A_chunk[d_state_idx] = A[batch_idx * seq_length * n_heads + seq_idx * n_heads + head_idx];
                    }
                    __syncthreads();
                    
                    // Precompute chunk base cumsum
                    float chunk_base_cumsum = A_cumsum[c];
                    
                    // Register for local cumsum to avoid repeated shared memory access
                    float local_cumsum = chunk_base_cumsum;
                    
                    // Process each position in the chunk with aggressive loop unrolling
                    #pragma unroll 8
                    for (int l = 0; l < block_len; l++) {
                        int seq_idx = c * block_len + l;
                        
                        // Update local cumsum (add previous A value)
                        if (l > 0) {
                            local_cumsum += A_chunk[l-1];
                        }
                        
                        // Compute decay factor: exp(last_cumsum - local_cumsum - current_A)
                        float decay = FAST_EXP(last_cumsum - local_cumsum - A_chunk[l]);
                        
                        // Get B and X values with coalesced access
                        float b_val = B[
                            batch_idx * seq_length * n_heads * d_state + 
                            seq_idx * n_heads * d_state + 
                            head_idx * d_state + 
                            d_state_idx
                        ];
                        
                        float x_val = X[
                            batch_idx * seq_length * n_heads * d_head + 
                            seq_idx * n_heads * d_head + 
                            head_idx * d_head + 
                            d_head_idx
                        ];
                        
                        // Accumulate directly to final state
                        final_state += decay * b_val * x_val;
                    }
                    __syncthreads();
                }
                
                // Write output with coalesced access
                output[
                    batch_idx * n_heads * d_head * d_state + 
                    head_idx * d_head * d_state + 
                    d_head_idx * d_state + 
                    d_state_idx
                ] = final_state;
            }
        }
        
        torch::Tensor mamba_final_state_cuda(
            torch::Tensor X,
            torch::Tensor A,
            torch::Tensor B,
            torch::Tensor initial_states,
            int batch_size,
            int seq_length,
            int n_heads,
            int d_head,
            int d_state,
            int block_len) {
            
            // Create output tensor
            auto options = torch::TensorOptions()
                .dtype(X.dtype())
                .device(X.device());
            
            auto output = torch::empty({batch_size, n_heads, d_head, d_state}, options);
            
            // Define block and grid dimensions - optimized for the specific problem
            // Each thread processes multiple elements in d_head dimension
            dim3 threads(32, 4);  // 32x4 = 128 threads per block, each thread processes 16 elements in d_head
            dim3 blocks(batch_size, n_heads);
            
            // Calculate shared memory size with padding to avoid bank conflicts
            int n_chunks = seq_length / block_len;
            int shared_mem_size = (block_len + 32 + n_chunks + 1 + 32) * sizeof(float);
            
            // Launch kernel
            AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "mamba_final_state_kernel", ([&] {
                mamba_final_state_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
                    X.data_ptr<scalar_t>(),
                    A.data_ptr<scalar_t>(),
                    B.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    initial_states.defined() ? initial_states.data_ptr<scalar_t>() : nullptr,
                    batch_size,
                    seq_length,
                    n_heads,
                    d_head,
                    d_state,
                    block_len
                );
            }));
            
            return output;
        }
        
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("mamba_final_state", &mamba_final_state_cuda, "Mamba final state computation (CUDA)");
        }
        """
        
        return load_inline(
            name="mamba_cuda_optimized",
            cpp_sources="",
            cuda_sources=cuda_source,
            functions=["mamba_final_state"],
            verbose=False
        )
    
    def forward_cuda(self, X, initial_states=None):
        """Forward pass using optimized CUDA kernel."""
        # Ensure tensors are contiguous
        X = X.contiguous()
        A = self.A.contiguous()
        B = self.B.contiguous()
        
        # Create initial states if not provided
        if initial_states is None:
            initial_states = torch.zeros(
                self.batch_size, self.n_heads, self.d_head, self.d_state,
                device=X.device, dtype=X.dtype
            )
        else:
            # Reshape initial_states if needed
            initial_states = initial_states.view(self.batch_size, self.n_heads, self.d_head, self.d_state).contiguous()
        
        # Call CUDA kernel
        return self.cuda_kernel.mamba_final_state(
            X, A, B, initial_states,
            self.batch_size, self.seq_length, self.n_heads, 
            self.d_head, self.d_state, self.block_len
        )
    
    def forward_pytorch(self, X, initial_states=None):
        """Optimized PyTorch implementation as fallback."""
        # Ensure tensors are contiguous for better performance
        X = X.contiguous()
        
        # Get number of chunks
        n_chunks = self.seq_length // self.block_len
        
        # Reshape tensors directly with view for better performance
        X_blocks = X.view(self.batch_size, n_chunks, self.block_len, self.n_heads, self.d_head)
        A_blocks = self.A.view(self.batch_size, n_chunks, self.block_len, self.n_heads)
        B_blocks = self.B.view(self.batch_size, n_chunks, self.block_len, self.n_heads, self.d_state)
        
        # Optimize A_blocks computation
        A_blocks_rearranged = A_blocks.permute(0, 3, 1, 2)  # [b, h, c, l]
        A_cumsum = torch.cumsum(A_blocks_rearranged, dim=-1)
        A_cumsum_last = A_cumsum[:, :, :, -1:]  # Last value of each chunk
        
        # Compute decay states: exp(A_cumsum_last - A_cumsum)
        decay_states = torch.exp(A_cumsum_last - A_cumsum)
        
        # Reshape decay_states for efficient computation with B_blocks
        decay_states_reshaped = decay_states.permute(0, 2, 3, 1)  # [b, c, l, h]
        
        # Apply decay to B_blocks using broadcasting
        B_decay = B_blocks * decay_states_reshaped.unsqueeze(-1)  # [b, c, l, h, n]
        
        # Optimize the states computation using batch matrix multiplication
        # [b, c, l, h, n] -> [b, c, h, n, l]
        B_decay_transposed = B_decay.permute(0, 1, 3, 4, 2)
        
        # [b, c, l, h, p] -> [b, c, h, l, p]
        X_blocks_reshaped = X_blocks.permute(0, 1, 3, 2, 4)
        
        # Perform batch matrix multiplication
        # [b, c, h, n, l] @ [b, c, h, l, p] -> [b, c, h, n, p]
        states = torch.matmul(B_decay_transposed, X_blocks_reshaped)
        
        # Transpose to get [b, c, h, p, n]
        states = states.permute(0, 1, 2, 4, 3)
        
        # Create initial states if not provided
        if initial_states is None:
            initial_states = torch.zeros(
                self.batch_size, 1, self.n_heads, self.d_head, self.d_state,
                device=X.device, dtype=X.dtype
            )
        else:
            # Reshape initial_states if needed
            initial_states = initial_states.view(self.batch_size, 1, self.n_heads, self.d_head, self.d_state)
        
        # Concatenate initial states with computed states
        states = torch.cat([initial_states, states], dim=1)
        
        # Optimize decay_chunk computation - only compute what's needed for final state
        A_cumsum_last_squeezed = A_cumsum_last.squeeze(-1)  # [b, h, c]
        A_padded = F.pad(A_cumsum_last_squeezed, (1, 0))  # [b, h, c+1]
        
        # Compute the last row of decay_chunk which is needed for the final state
        x_cumsum = torch.cumsum(A_padded, dim=-1)  # [b, h, c+1]
        last_cumsum = x_cumsum[:, :, -1].unsqueeze(-1)  # [b, h, 1]
        decay_last_row = torch.exp(last_cumsum - x_cumsum)  # [b, h, c+1]
        
        # Reshape for efficient broadcasting
        decay_last_row = decay_last_row.unsqueeze(-1).unsqueeze(-1)  # [b, h, c+1, 1, 1]
        
        # Rearrange states for efficient computation
        states_transposed = states.permute(0, 2, 1, 3, 4)  # [b, h, c+1, p, n]
        
        # Compute final state: sum(decay_last_row * states_transposed, dim=2)
        final_state = (decay_last_row * states_transposed).sum(dim=2)  # [b, h, p, n]
        
        return final_state
    
    def forward(self, X, initial_states=None):
        """
        Forward pass implementing the SSD operation.
        
        :param X: Input tensor of shape (batch, length, n_heads, d_head)
        :param initial_states: Optional initial states
        :return: Final state
        """
        if self.cuda_kernel is not None and X.is_cuda:
            try:
                return self.forward_cuda(X, initial_states)
            except Exception as e:
                print(f"CUDA kernel failed, falling back to PyTorch: {e}")
                return self.forward_pytorch(X, initial_states)
        else:
            return self.forward_pytorch(X, initial_states)

# Test parameters
batch_size = 16
seq_length = 128
n_heads = 8
d_head = 64
d_state = 16
block_len = 64

def get_inputs():
    return [torch.randn(batch_size, seq_length, n_heads, d_head)]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]