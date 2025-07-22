import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to `dropout`
        """
        super(ModelNew, self).__init__()
        # Initialize hidden state with random values
        self.h0 = torch.randn((num_layers, batch_size, hidden_size))
        self.c0 = torch.randn((num_layers, batch_size, hidden_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Cache for device tensors
        self._h0_device = None
        self._c0_device = None
        self._last_device = None
        
        # For CUDA graph optimization
        self._graph = None
        self._static_input = None
        self._static_h0 = None
        self._static_c0 = None
        self._static_output = None
        
        # Pin memory for faster transfers
        if torch.cuda.is_available():
            self.h0 = self.h0.pin_memory()
            self.c0 = self.c0.pin_memory()
        
        # Stream for computation
        self.stream = None
        
        # Warmup flag
        self._warmup_done = False
    
    def _ensure_device_tensors(self, device):
        """Ensure hidden states are on the correct device and cached"""
        if self._h0_device is None or self._last_device != device:
            self._h0_device = self.h0.to(device, non_blocking=True)
            self._c0_device = self.c0.to(device, non_blocking=True)
            self._last_device = device
        
        return self._h0_device, self._c0_device
    
    def _maybe_capture_graph(self, x, h0, c0):
        """Capture computation graph for repeated execution if possible"""
        if not torch.cuda.is_available() or not hasattr(torch.cuda, 'graphs'):
            return None
        
        # Check if we need to create a new graph
        if self._graph is None:
            try:
                # Create static tensors for graph capture
                self._static_input = torch.zeros_like(x)
                self._static_h0 = torch.zeros_like(h0)
                self._static_c0 = torch.zeros_like(c0)
                self._static_output = torch.zeros_like(h0)
                
                # Capture the graph
                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph):
                    self._static_input.copy_(x)
                    self._static_h0.copy_(h0)
                    self._static_c0.copy_(c0)
                    
                    # Forward pass
                    out, state = self.lstm(self._static_input, 
                                          (self._static_h0, self._static_c0))
                    
                    # Store only the hidden state (h) as required by the reference implementation
                    self._static_output.copy_(state[0])
                
            except Exception:
                # If graph capture fails, return None to fall back to standard execution
                self._graph = None
                return None
        
        # Copy input data to static tensors
        self._static_input.copy_(x)
        self._static_h0.copy_(h0)
        self._static_c0.copy_(c0)
        
        # Replay the graph
        self._graph.replay()
        
        # Return the result
        return self._static_output
    
    def _warmup(self, x, h0, c0):
        """Perform warmup to initialize CUDA kernels and caches"""
        if not x.is_cuda or self._warmup_done:
            return
            
        # Run a few iterations to warm up CUDA kernels
        with torch.no_grad():
            for _ in range(2):
                self.lstm(x, (h0, c0))
                
        # Try to capture the graph once during warmup
        self._maybe_capture_graph(x, h0, c0)
        
        self._warmup_done = True
    
    def forward(self, x):
        """
        Forward pass through the LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :return: The output tensor, shape (batch_size, sequence_length, output_size)
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Get device tensors for hidden states
        h0, c0 = self._ensure_device_tensors(x.device)
        
        # Try to use CUDA graph if possible (for GPU execution)
        if x.is_cuda:
            # Initialize stream if needed
            if self.stream is None:
                self.stream = torch.cuda.Stream(x.device)
            
            # Perform warmup if needed
            if not self._warmup_done:
                self._warmup(x, h0, c0)
            
            with torch.cuda.stream(self.stream):
                # Try using CUDA graph
                graph_result = self._maybe_capture_graph(x, h0, c0)
                if graph_result is not None:
                    return graph_result
                
                # Fallback to standard execution if graph capture/replay failed
                out, state = self.lstm(x, (h0, c0))
                return state[0]
            
            # Wait for computation to complete before returning
            torch.cuda.current_stream(x.device).wait_stream(self.stream)
            
            return state[0]
        else:
            # CPU execution path (no CUDA graphs or streams)
            out, state = self.lstm(x, (h0, c0))
            return state[0]

# Test code
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    return [torch.randn(batch_size, sequence_length, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]