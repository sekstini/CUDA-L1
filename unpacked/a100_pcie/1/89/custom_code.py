import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    An optimized model that performs a cumulative sum (prefix sum) operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the optimized Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        # Pre-allocate buffer for output
        self._buffer = None
        self._buffer_shape = None
        self._buffer_device = None
        # CUDA stream for asynchronous execution
        self._stream = None
        # Warmup flag
        self._warmup_done = False

    def forward(self, x):
        """
        Forward pass for the optimized Scan model, computing the cumulative sum along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative sum along `dim`.
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # For CUDA tensors, use our optimized path
        if x.is_cuda:
            # Check if device has changed and update stream if needed
            if self._buffer_device != x.device:
                self._stream = torch.cuda.Stream(device=x.device)
                self._buffer_device = x.device
                # Force buffer reallocation on device change
                self._buffer = None
                self._warmup_done = False
            
            # Check if we need to allocate or resize our buffer
            if self._buffer is None or self._buffer_shape != x.shape:
                self._buffer = torch.empty_like(x)
                self._buffer_shape = x.shape
                self._warmup_done = False
            
            # Use our stream for all operations
            with torch.cuda.stream(self._stream):
                # Perform warmup calculation if this is the first run
                # This helps initialize CUDA resources and can improve subsequent runs
                if not self._warmup_done:
                    torch.cumsum(x, dim=self.dim, out=self._buffer)
                    self._warmup_done = True
                
                # Use PyTorch's native cumsum with pre-allocated buffer
                torch.cumsum(x, dim=self.dim, out=self._buffer)
            
            return self._buffer
        else:
            # For CPU tensors, also use buffer if available
            if self._buffer is None or self._buffer_shape != x.shape or self._buffer_device is not None:
                self._buffer = torch.empty_like(x)
                self._buffer_shape = x.shape
                self._buffer_device = None
            
            # Use out parameter to avoid additional allocation
            torch.cumsum(x, dim=self.dim, out=self._buffer)
            return self._buffer

# Define input dimensions and parameters
batch_size = 128
input_shape = (4000,)  # Example shape
dim = 1

def get_inputs():
    """
    Generates random inputs for testing the Scan model.

    Returns:
        list: A list containing a single randomly generated tensor with shape 
              (batch_size, *input_shape).
    """
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    """
    Returns the initialization parameters for the Scan model.

    Returns:
        list: A list containing the `dim` parameter for model initialization.
    """
    return [dim]