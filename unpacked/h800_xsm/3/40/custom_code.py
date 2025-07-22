import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers (default: 1)
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh (default: True)
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) (default: False)
        """
        super(ModelNew, self).__init__()
        
        # Create the GRU layer with the same parameters as the reference implementation
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=False)
        self.h0 = torch.randn((num_layers, batch_size, hidden_size))
        
        # Cache for device-specific hidden state
        self._cached_h0 = None
        self._last_device = None
        self._last_dtype = None
        
        # CUDA graph related attributes
        self._cuda_graph = None
        self._static_input = None
        self._static_h0 = None
        self._static_output = None
        self._graph_captured = False
        
        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True
    
    def _ensure_h0_on_device(self, device, dtype):
        """Ensure hidden state is on the correct device and has the right dtype"""
        if self._cached_h0 is None or self._last_device != device or self._last_dtype != dtype:
            self._cached_h0 = self.h0.to(device=device, dtype=dtype, non_blocking=True)
            if not self._cached_h0.is_contiguous():
                self._cached_h0 = self._cached_h0.contiguous()
            self._last_device = device
            self._last_dtype = dtype
        return self._cached_h0
    
    def _can_use_cuda_graph(self, x):
        """Check if we can use CUDA graph for this input"""
        if not torch.cuda.is_available() or not x.is_cuda:
            return False
        
        # Check CUDA capabilities - CUDA graphs require compute capability >= 7.0
        try:
            device_props = torch.cuda.get_device_properties(x.device)
            if device_props.major < 7:
                return False
            return True
        except:
            return False
    
    def _capture_cuda_graph(self, x):
        """Capture CUDA graph for faster execution"""
        try:
            # Create static tensors for CUDA graph
            self._static_input = torch.zeros_like(x, requires_grad=False)
            self._static_h0 = torch.zeros_like(self._cached_h0, requires_grad=False)
            self._static_output = torch.zeros_like(self._cached_h0, requires_grad=False)
            
            # Warm up to ensure cuDNN selects optimal algorithms
            for _ in range(3):
                _, h_n = self.gru(x, self._cached_h0)
            
            # Capture the graph
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self._cuda_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._cuda_graph):
                    self._static_input.copy_(x)
                    self._static_h0.copy_(self._cached_h0)
                    _, h_n = self.gru(self._static_input, self._static_h0)
                    self._static_output.copy_(h_n)
            
            torch.cuda.current_stream().wait_stream(stream)
            self._graph_captured = True
            return True
        except Exception:
            # Clean up if capture failed
            self._static_input = None
            self._static_h0 = None
            self._static_output = None
            self._cuda_graph = None
            self._graph_captured = False
            return False
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, 
                 otherwise (batch_size, seq_len, input_size)
        :return: h_n: The hidden state for t = seq_len, shape (num_layers * num_directions, batch_size, hidden_size)
        """
        # Ensure input is contiguous for better memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Ensure hidden state is on the correct device
        self._cached_h0 = self._ensure_h0_on_device(x.device, x.dtype)
        
        # Try to use CUDA graph for better performance
        if self._can_use_cuda_graph(x):
            # Capture graph if not already captured
            if not self._graph_captured:
                self._capture_cuda_graph(x)
            
            # If graph is captured successfully, use it
            if self._graph_captured:
                try:
                    self._static_input.copy_(x)
                    self._static_h0.copy_(self._cached_h0)
                    self._cuda_graph.replay()
                    return self._static_output.clone()
                except Exception:
                    # Fall back to standard execution if replay fails
                    pass
        
        # Standard execution path
        _, h_n = self.gru(x, self._cached_h0)
        return h_n

# Hyperparameters - copied exactly from reference implementation
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    return [torch.randn(seq_len, batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]