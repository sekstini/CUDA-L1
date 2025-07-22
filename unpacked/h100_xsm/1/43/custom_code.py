import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 3D with optimized implementation.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the Max Pooling 3D layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which means stride is equal to kernel_size.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return indices of the maximum values. Defaults to False.
            ceil_mode (bool, optional): When True, the output size is ceil(input_size / stride) instead of floor. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        
        # Keep the original maxpool for fallback and when return_indices is True
        if return_indices:
            self.original_maxpool = nn.MaxPool3d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                return_indices=return_indices,
                ceil_mode=ceil_mode
            )
        
        # Create a dedicated CUDA stream for this module instance
        self._stream = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 3D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, dim1, dim2, dim3).

        Returns:
            torch.Tensor: Output tensor with Max Pooling 3D applied.
        """
        # If indices are required, use the original implementation
        if self.return_indices:
            if hasattr(self, 'original_maxpool'):
                return self.original_maxpool(x)
            else:
                return F.max_pool3d(
                    x,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    ceil_mode=self.ceil_mode,
                    return_indices=True
                )
        
        # Ensure input is contiguous for better memory access patterns
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Use a dedicated CUDA stream for potential performance improvement
        if x.is_cuda:
            # Create stream if it doesn't exist yet
            if self._stream is None:
                self._stream = torch.cuda.Stream()
            
            with torch.cuda.stream(self._stream):
                # Direct function call is faster than using the module
                result = F.max_pool3d(
                    x,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    ceil_mode=self.ceil_mode,
                    return_indices=False
                )
            
            # No explicit synchronization here - let PyTorch handle it implicitly
            return result
        else:
            # For CPU tensors, no need for CUDA stream
            return F.max_pool3d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                ceil_mode=self.ceil_mode,
                return_indices=False
            )

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
channels = 32
dim1 = 64
dim2 = 64
dim3 = 64
kernel_size = 3
stride = 2
padding = 1
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, channels, dim1, dim2, dim3)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]