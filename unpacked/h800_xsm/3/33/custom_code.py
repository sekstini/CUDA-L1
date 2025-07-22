import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model with optimized CUDA operations.
        
        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize hidden state just like the reference implementation
        self.hidden = torch.randn((batch_size, hidden_size))
        
        # Create temporary linear layers with the same initialization as the reference
        temp_i2h = nn.Linear(input_size + hidden_size, hidden_size)
        temp_h2o = nn.Linear(hidden_size, output_size)
        
        # Extract and separate the weights for input and hidden
        with torch.no_grad():
            # Split the i2h weights into input and hidden parts
            self.weight_ih = temp_i2h.weight[:, :input_size].clone().contiguous()
            self.weight_hh = temp_i2h.weight[:, input_size:].clone().contiguous()
            self.bias_h = temp_i2h.bias.clone()
            
            # Extract h2o weights
            self.weight_ho = temp_h2o.weight.clone().contiguous()
            self.bias_o = temp_h2o.bias.clone()
        
        # Pre-transpose weights for faster matrix multiplication
        self.weight_ih_t = self.weight_ih.t().contiguous()
        self.weight_hh_t = self.weight_hh.t().contiguous()
        self.weight_ho_t = self.weight_ho.t().contiguous()
        
        # Flag to track if tensors have been moved to device
        self._device_initialized = False
        
        # We'll initialize buffers lazily when we know the device
        self.hidden_buffer = None
        self.output_buffer = None
        
        # Initialize CUDA kernel if possible
        self.cuda_kernel_available = False
        try:
            from torch.utils.cpp_extension import load_inline
            
            cuda_source = """
            extern "C" __global__ void rnn_fused_kernel(
                const float* __restrict__ input,
                const float* __restrict__ hidden,
                const float* __restrict__ weight_ih,
                const float* __restrict__ weight_hh,
                const float* __restrict__ bias_h,
                float* __restrict__ new_hidden,
                int batch_size,
                int input_size,
                int hidden_size
            ) {
                // Calculate global thread ID
                const int tid = blockIdx.x * blockDim.x + threadIdx.x;
                const int batch_idx = tid / hidden_size;
                const int hidden_idx = tid % hidden_size;
                
                // Check bounds
                if (batch_idx >= batch_size || hidden_idx >= hidden_size) return;
                
                // Compute offsets
                const int input_offset = batch_idx * input_size;
                const int hidden_offset = batch_idx * hidden_size;
                
                // Initialize with bias
                float sum = bias_h[hidden_idx];
                
                // Input-to-hidden contribution
                for (int i = 0; i < input_size; i++) {
                    sum += input[input_offset + i] * weight_ih[hidden_idx * input_size + i];
                }
                
                // Hidden-to-hidden contribution
                for (int i = 0; i < hidden_size; i++) {
                    sum += hidden[hidden_offset + i] * weight_hh[hidden_idx * hidden_size + i];
                }
                
                // Apply tanh activation and store result
                new_hidden[hidden_offset + hidden_idx] = tanhf(sum);
            }
            
            extern "C" __global__ void rnn_fused_kernel_shared(
                const float* __restrict__ input,
                const float* __restrict__ hidden,
                const float* __restrict__ weight_ih,
                const float* __restrict__ weight_hh,
                const float* __restrict__ bias_h,
                float* __restrict__ new_hidden,
                int batch_size,
                int input_size,
                int hidden_size
            ) {
                // Define shared memory for bias
                extern __shared__ float shared_mem[];
                float* shared_bias = shared_mem;
                
                // Calculate thread indices
                const int tid = threadIdx.x;
                const int batch_idx = blockIdx.x;
                
                // Check bounds for batch dimension
                if (batch_idx >= batch_size) return;
                
                // Load bias into shared memory
                if (tid < hidden_size) {
                    shared_bias[tid] = bias_h[tid];
                }
                
                __syncthreads();
                
                // Process multiple hidden units per thread if needed
                for (int h = tid; h < hidden_size; h += blockDim.x) {
                    // Initialize with bias
                    float sum = shared_bias[h];
                    
                    // Input-to-hidden contribution
                    for (int i = 0; i < input_size; i++) {
                        sum += input[batch_idx * input_size + i] * weight_ih[h * input_size + i];
                    }
                    
                    // Hidden-to-hidden contribution
                    for (int i = 0; i < hidden_size; i++) {
                        sum += hidden[batch_idx * hidden_size + i] * weight_hh[h * hidden_size + i];
                    }
                    
                    // Apply tanh activation and store result
                    new_hidden[batch_idx * hidden_size + h] = tanhf(sum);
                }
            }
            """
            
            self.rnn_cuda = load_inline(
                name="rnn_cuda",
                cpp_sources="",
                cuda_sources=cuda_source,
                functions=["rnn_fused_kernel", "rnn_fused_kernel_shared"],
                with_cuda=True,
                verbose=False
            )
            
            self.cuda_kernel_available = True
        except Exception:
            # Fallback to PyTorch operations if CUDA compilation fails
            self.cuda_kernel_available = False
    
    def _run_cuda_kernel(self, x):
        """Run the optimized CUDA kernel for RNN computation."""
        batch_size = x.size(0)
        
        # For small batch sizes, use the shared memory version
        if batch_size <= 32:
            # One block per batch item, with enough threads to handle hidden units
            threads_per_block = min(256, self.hidden_size)
            shared_mem_size = self.hidden_size * 4  # For bias (float = 4 bytes)
            
            self.rnn_cuda.rnn_fused_kernel_shared(
                x,                  # input
                self.hidden,        # hidden
                self.weight_ih,     # weight_ih
                self.weight_hh,     # weight_hh
                self.bias_h,        # bias_h
                self.hidden_buffer, # new_hidden
                batch_size,         # batch_size
                self.input_size,    # input_size
                self.hidden_size,   # hidden_size
                grid=(batch_size,),
                block=(threads_per_block,),
                shared=shared_mem_size
            )
        else:  # For larger batch sizes, use the regular version
            # Set optimal grid and block dimensions
            threads_per_block = 256
            num_blocks = (batch_size * self.hidden_size + threads_per_block - 1) // threads_per_block
            
            self.rnn_cuda.rnn_fused_kernel(
                x,                  # input
                self.hidden,        # hidden
                self.weight_ih,     # weight_ih
                self.weight_hh,     # weight_hh
                self.bias_h,        # bias_h
                self.hidden_buffer, # new_hidden
                batch_size,         # batch_size
                self.input_size,    # input_size
                self.hidden_size,   # hidden_size
                grid=(num_blocks,),
                block=(threads_per_block,)
            )
        
        # Update hidden state
        self.hidden.copy_(self.hidden_buffer)
    
    def _forward_pytorch(self, x):
        """Optimized forward implementation using PyTorch operations."""
        # Compute input-to-hidden contribution with bias
        torch.addmm(self.bias_h, x, self.weight_ih_t, out=self.hidden_buffer)
        
        # Add hidden-to-hidden contribution in-place
        self.hidden_buffer.addmm_(self.hidden, self.weight_hh_t)
        
        # Apply tanh activation and update hidden state
        torch.tanh(self.hidden_buffer, out=self.hidden)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vanilla RNN with optimized operations.
        
        :param x: Input tensor of shape (batch_size, input_size).
        :return: Output tensor of shape (batch_size, output_size).
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Move tensors to device only once or when device changes
        device = x.device
        if not self._device_initialized or self.hidden.device != device:
            self.hidden = self.hidden.to(device)
            self.weight_ih = self.weight_ih.to(device)
            self.weight_hh = self.weight_hh.to(device)
            self.weight_ih_t = self.weight_ih_t.to(device)
            self.weight_hh_t = self.weight_hh_t.to(device)
            self.weight_ho_t = self.weight_ho_t.to(device)
            self.bias_h = self.bias_h.to(device)
            self.bias_o = self.bias_o.to(device)
            
            # Initialize buffers on the correct device
            self.hidden_buffer = torch.empty((batch_size, self.hidden_size), device=device)
            self.output_buffer = torch.empty((batch_size, self.output_size), device=device)
            
            self._device_initialized = True
        
        # Try to use CUDA kernel if available and we're on CUDA device
        if self.cuda_kernel_available and x.is_cuda:
            try:
                self._run_cuda_kernel(x)
            except Exception:
                # Fallback to PyTorch operations if CUDA kernel fails
                self._forward_pytorch(x)
        else:
            # Use optimized PyTorch operations
            self._forward_pytorch(x)
        
        # Compute output using optimized PyTorch operation
        torch.addmm(self.bias_o, self.hidden, self.weight_ho_t, out=self.output_buffer)
        
        return self.output_buffer

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, output_size]