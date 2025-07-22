import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Create the GRU layer with the exact same parameters
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
        
        # Register h0 as a buffer to ensure it's moved to the correct device with the model
        self.register_buffer('h0', torch.randn((num_layers * 2, batch_size, hidden_size)))
        
        # Store configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        
        # Optimize parameter layout during initialization
        self.gru.flatten_parameters()
        
        # For CUDA graph management
        self._current_device = None
        self._graph_created = False
        self._cuda_graph = None
        self._static_input = None
        self._static_output = None
        self._static_h_n = None
        self._input_shape = None
        
        # Create CUDA streams for potential parallel execution
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, 
                 otherwise (batch_size, seq_len, input_size)
        :return: output: The output features from the last layer of the GRU
        """
        # Check if device has changed
        if self._current_device != x.device:
            self._current_device = x.device
            self.h0 = self.h0.to(x.device)
            self._graph_created = False
        
        # Ensure input is contiguous for better memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Use CUDA graph if available and input is on CUDA
        if x.is_cuda and torch.cuda.is_available():
            # Check if input shape has changed
            if self._input_shape != x.shape:
                self._input_shape = x.shape
                self._graph_created = False
            
            with torch.cuda.stream(self.stream):
                if not self._graph_created:
                    # First time or device/shape changed - run normally and capture graph
                    self._static_input = x.clone()
                    
                    # Determine output shape based on input shape and GRU configuration
                    if self.batch_first:
                        output_shape = (x.size(0), x.size(1), self.hidden_size * 2)
                    else:
                        output_shape = (x.size(0), x.size(1), self.hidden_size * 2)
                    
                    self._static_output = torch.empty(
                        output_shape,
                        device=x.device, 
                        dtype=x.dtype
                    )
                    
                    self._static_h_n = torch.empty(
                        (self.num_layers * 2, x.size(1) if not self.batch_first else x.size(0), self.hidden_size),
                        device=x.device, 
                        dtype=x.dtype
                    )
                    
                    # Warmup pass
                    with torch.no_grad():
                        self.gru.flatten_parameters()
                        _ = self.gru(self._static_input, self.h0)
                    
                    # Capture graph
                    try:
                        self._cuda_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(self._cuda_graph):
                            self._static_output, self._static_h_n = self.gru(self._static_input, self.h0)
                        self._graph_created = True
                    except Exception as e:
                        # Fall back to normal execution if graph capture fails
                        self._graph_created = False
                        print(f"CUDA graph capture failed: {e}")
                
                if self._graph_created:
                    # Copy input to static tensor
                    self._static_input.copy_(x)
                    # Replay graph
                    self._cuda_graph.replay()
                    # Return output from static tensor
                    return self._static_output
                else:
                    # Normal execution
                    self.gru.flatten_parameters()
                    output, _ = self.gru(x, self.h0)
                    return output
        else:
            # For CPU, just run the standard implementation
            self.gru.flatten_parameters()
            output, _ = self.gru(x, self.h0)
            return output

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