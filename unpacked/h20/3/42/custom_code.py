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
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
        self.h0 = torch.randn((num_layers * 2, batch_size, hidden_size))
        
        # CUDA graph optimization
        self.graph = None
        self.static_input = None
        self.static_h0 = None
        self.static_output = None
        self.stream = None
        self.input_shape = None
        self.graph_initialized = False
    
    def _initialize_cuda_graph(self, x):
        """Initialize CUDA graph with static tensors"""
        # Save input shape to detect changes
        self.input_shape = x.shape
        
        # Create dedicated CUDA stream for graph capture and execution
        if self.stream is None:
            self.stream = torch.cuda.Stream()
        
        with torch.cuda.stream(self.stream):
            # Create static input and output tensors
            self.static_input = torch.zeros_like(x, device='cuda').contiguous()
            self.static_h0 = self.h0.cuda().contiguous()
            self.static_output = torch.zeros((self.gru.num_layers * 2, x.size(1), self.gru.hidden_size), 
                                          device='cuda').contiguous()
            
            # Warm-up iteration to ensure stable performance
            self.static_input.copy_(x.cuda())
            _, _ = self.gru(self.static_input, self.static_h0)
            
            # Ensure all operations are completed before capturing graph
            torch.cuda.synchronize()
            
            # Capture graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                _, self.static_output = self.gru(self.static_input, self.static_h0)
        
        self.graph_initialized = True
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, 
                  otherwise (batch_size, seq_len, input_size)
        :return: h_n: The hidden state for t = seq_len, shape (num_layers * num_directions, batch_size, hidden_size)
        """
        # Move h0 to the same device as x
        self.h0 = self.h0.to(x.device)
        
        # If not on CUDA, just use standard execution
        if not x.is_cuda:
            _, h_n = self.gru(x, self.h0)
            return h_n
        
        # Ensure input is contiguous for better memory access
        x = x.contiguous()
        
        # Check if we need to initialize or reinitialize the graph
        if (not self.graph_initialized or 
            self.input_shape != x.shape or
            self.static_input is None):
            
            # Initialize graph - if it fails, fall back to standard execution
            try:
                self._initialize_cuda_graph(x)
            except Exception:
                _, h_n = self.gru(x, self.h0)
                return h_n
        
        # Use CUDA graph for execution
        with torch.cuda.stream(self.stream):
            self.static_input.copy_(x)
            self.graph.replay()
        
        # Return the static output directly
        return self.static_output

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    return [torch.randn(seq_len, batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]