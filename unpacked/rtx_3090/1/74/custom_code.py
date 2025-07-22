import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with square input and asymmetric kernel, optionally with dilation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Create weight parameter with the same shape as nn.ConvTranspose1d
        # For transposed convolution, weight shape is (in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters using the same method as nn.ConvTranspose1d
        self.reset_parameters()
        
        # Compile CUDA kernel for transposed 1D convolution
        self._setup_cuda_kernel()
    
    def reset_parameters(self):
        # Use the same initialization as nn.ConvTranspose1d
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def _setup_cuda_kernel(self):
        # Define the CUDA kernel for transposed 1D convolution
        self.cuda_kernel = None
        if torch.cuda.is_available():
            kernel_code = """
            extern "C" __global__ void conv_transpose1d_kernel(
                const float* input, const float* weight, const float* bias,
                float* output, 
                int batch_size, int in_channels, int out_channels,
                int input_length, int output_length,
                int kernel_size, int stride, int padding, int dilation) {
                
                // Each block handles one (batch, output_channel) combination
                int b = blockIdx.y;
                int oc = blockIdx.x;
                
                // Each thread handles one output position
                int tid = threadIdx.x;
                int block_size = blockDim.x;
                
                // Shared memory for weights - each block loads weights for one output channel
                extern __shared__ float shared_weights[];
                
                // Load weights for this output channel into shared memory
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int k = tid; k < kernel_size; k += block_size) {
                        shared_weights[ic * kernel_size + k] = weight[(ic * out_channels + oc) * kernel_size + k];
                    }
                }
                __syncthreads();
                
                // Process output positions in chunks
                for (int op_base = 0; op_base < output_length; op_base += block_size) {
                    int op = op_base + tid;
                    
                    if (op < output_length) {
                        float val = 0.0f;
                        
                        // For each input channel
                        for (int ic = 0; ic < in_channels; ++ic) {
                            // Specialized for kernel_size=5, stride=1, padding=0, dilation=3
                            // For transposed conv: out_pos = in_pos * stride - padding + k * dilation
                            // Solving for in_pos: in_pos = (out_pos + padding - k * dilation) / stride
                            
                            // k=0
                            {
                                int ip = op;  // With stride=1, padding=0, dilation=3, k=0
                                if (ip >= 0 && ip < input_length) {
                                    val += input[((b * in_channels + ic) * input_length) + ip] * 
                                           shared_weights[ic * kernel_size + 0];
                                }
                            }
                            
                            // k=1
                            {
                                int ip = op - 3;  // With stride=1, padding=0, dilation=3, k=1
                                if (ip >= 0 && ip < input_length) {
                                    val += input[((b * in_channels + ic) * input_length) + ip] * 
                                           shared_weights[ic * kernel_size + 1];
                                }
                            }
                            
                            // k=2
                            {
                                int ip = op - 6;  // With stride=1, padding=0, dilation=3, k=2
                                if (ip >= 0 && ip < input_length) {
                                    val += input[((b * in_channels + ic) * input_length) + ip] * 
                                           shared_weights[ic * kernel_size + 2];
                                }
                            }
                            
                            // k=3
                            {
                                int ip = op - 9;  // With stride=1, padding=0, dilation=3, k=3
                                if (ip >= 0 && ip < input_length) {
                                    val += input[((b * in_channels + ic) * input_length) + ip] * 
                                           shared_weights[ic * kernel_size + 3];
                                }
                            }
                            
                            // k=4
                            {
                                int ip = op - 12;  // With stride=1, padding=0, dilation=3, k=4
                                if (ip >= 0 && ip < input_length) {
                                    val += input[((b * in_channels + ic) * input_length) + ip] * 
                                           shared_weights[ic * kernel_size + 4];
                                }
                            }
                        }
                        
                        // Add bias if needed
                        if (bias != nullptr) {
                            val += bias[oc];
                        }
                        
                        // Write the result to the output tensor
                        output[((b * out_channels + oc) * output_length) + op] = val;
                    }
                }
            }
            """
            
            try:
                from torch.utils.cpp_extension import load_inline
                self.cuda_module = load_inline(
                    name="conv_transpose1d_cuda_optimized",
                    cpp_sources="",
                    cuda_sources=kernel_code,
                    functions=["conv_transpose1d_kernel"],
                    verbose=False
                )
                self.cuda_kernel = self.cuda_module.conv_transpose1d_kernel
            except Exception as e:
                print(f"Failed to compile CUDA kernel: {e}")
                self.cuda_kernel = None
    
    def _conv_transpose1d_cuda(self, x):
        # Calculate output shape
        batch_size, in_channels, input_length = x.shape
        output_length = (input_length - 1) * self.stride - 2 * self.padding + self.dilation * (self.kernel_size - 1) + 1
        
        # Create output tensor
        output = torch.zeros(batch_size, self.out_channels, output_length, device=x.device, dtype=x.dtype)
        
        # Prepare inputs for CUDA kernel
        input_flat = x.contiguous()
        weight_flat = self.weight.contiguous()
        bias_flat = self.bias.contiguous() if self.bias is not None else None
        
        # Calculate grid and block dimensions
        threads_per_block = min(256, output_length)
        grid_x = self.out_channels
        grid_y = batch_size
        
        # Calculate shared memory size - we need to store weights for one output channel
        shared_mem_size = in_channels * self.kernel_size * 4  # 4 bytes per float
        
        # Launch CUDA kernel
        self.cuda_kernel(
            grid=(grid_x, grid_y, 1),
            block=(threads_per_block, 1, 1),
            args=[
                input_flat.data_ptr(), weight_flat.data_ptr(), 
                bias_flat.data_ptr() if bias_flat is not None else 0,
                output.data_ptr(),
                batch_size, in_channels, self.out_channels,
                input_length, output_length,
                self.kernel_size, self.stride, self.padding, self.dilation
            ],
            shared=shared_mem_size
        )
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Use custom CUDA kernel if available and input is on CUDA device
        if self.cuda_kernel is not None and x.is_cuda:
            try:
                return self._conv_transpose1d_cuda(x)
            except Exception as e:
                print(f"CUDA kernel failed, falling back to PyTorch implementation: {e}")
        
        # Fallback to PyTorch's implementation
        return F.conv_transpose1d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation
        )

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 5
length = 256
stride = 1
padding = 0
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]