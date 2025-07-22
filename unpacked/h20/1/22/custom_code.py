import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Simple model that performs a Tanh activation with highly optimized memory usage.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # Cache for output tensor and input properties
        self._output = None
        self._last_input_ptr = None  # For ultra-fast identity check
        self._last_shape = None
        self._last_device = None
        self._last_dtype = None
        self._last_input_was_contiguous = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation to the input tensor with optimized execution.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        # Ultra-fast path: exact same tensor object (by memory address)
        current_input_ptr = x.data_ptr()
        if current_input_ptr == self._last_input_ptr and self._output is not None:
            # We can completely skip all property checks
            torch.tanh(x, out=self._output)
            return self._output

        # Check if we need to reallocate the output tensor
        need_reallocation = (
            self._output is None or 
            x.shape != self._last_shape or 
            x.device != self._last_device or 
            x.dtype != self._last_dtype
        )
        
        # Check if input is contiguous - needed for optimal memory access
        input_is_contiguous = x.is_contiguous()
        input_tensor = x if input_is_contiguous else x.contiguous()
        
        # Update current_input_ptr if we created a new contiguous tensor
        if not input_is_contiguous:
            current_input_ptr = input_tensor.data_ptr()
        
        # Reallocate output if needed
        if need_reallocation:
            self._output = torch.empty_like(input_tensor)
            self._last_shape = x.shape
            self._last_device = x.device
            self._last_dtype = x.dtype
        
        # Update cached input pointer and contiguity status
        self._last_input_ptr = current_input_ptr
        self._last_input_was_contiguous = input_is_contiguous
        
        # Compute tanh directly into pre-allocated output
        torch.tanh(input_tensor, out=self._output)
        
        return self._output

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed