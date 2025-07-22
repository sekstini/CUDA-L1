import torch
import torch.nn as nn
import collections

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model with optimized CUDA execution.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        """
        super(ModelNew, self).__init__()
        
        # Register hidden state and cell state as buffers to ensure they're moved to the right device
        self.register_buffer('h0', torch.randn((num_layers, batch_size, hidden_size)))
        self.register_buffer('c0', torch.randn((num_layers, batch_size, hidden_size)))
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
        # LRU cache for CUDA graphs with a maximum size to prevent memory bloat
        self.graph_cache = collections.OrderedDict()
        self.max_cache_size = 3  # Optimal size based on empirical results
        
        # Fast path tracking for the most recent input
        self.last_key = None
        self.last_static_input = None
        self.last_static_output = None
        self.last_graph = None
        
        # Create a dedicated stream for graph execution
        self.stream = None
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        
        # Try to script the LSTM for better performance
        self.scripted_lstm = None
        if not torch.jit.is_scripting() and torch.cuda.is_available():
            try:
                self.scripted_lstm = torch.jit.script(self.lstm)
            except Exception:
                pass
                
        # Pre-compile critical paths
        self._warmup()
    
    def _warmup(self):
        """Pre-compile critical paths during initialization"""
        if not torch.cuda.is_available():
            return
            
        try:
            with torch.no_grad(), torch.cuda.stream(self.stream):
                # Create a sample input on GPU
                device = torch.device('cuda')
                sample_input = torch.zeros((batch_size, sequence_length, input_size), device=device)
                
                # Move hidden states to GPU
                h0 = self.h0.to(device)
                c0 = self.c0.to(device)
                
                # Warm up LSTM execution
                for _ in range(3):
                    _, (_, _) = self.lstm(sample_input, (h0, c0))
                
                # Synchronize to ensure warmup is complete
                self.stream.synchronize()
        except Exception:
            # Silently ignore warmup failures
            pass
    
    def _get_cache_key(self, x):
        """Create an efficient cache key based on input properties"""
        return (x.shape, x.device.index if x.is_cuda else -1)
    
    def _create_cuda_graph(self, x):
        """Create a CUDA graph for the given input configuration"""
        if not torch.cuda.is_available() or not x.is_cuda:
            return None, None, None
        
        try:
            # Create static input tensor
            static_input = x.clone()
            
            # Ensure hidden states are on the correct device and contiguous
            h0 = self.h0.to(x.device, non_blocking=True).contiguous()
            c0 = self.c0.to(x.device, non_blocking=True).contiguous()
            
            # Determine optimal warmup iterations based on input size
            tensor_size = x.numel() * x.element_size()
            warmup_iterations = max(3, min(5, tensor_size // (1024 * 1024) + 2))
            
            # Use our dedicated stream for graph operations
            with torch.cuda.stream(self.stream):
                # Warmup runs to ensure stable execution
                for _ in range(warmup_iterations):
                    with torch.no_grad():
                        _, (_, _) = self.lstm(static_input, (h0, c0))
                
                # Synchronize before graph capture
                self.stream.synchronize()
                
                # Capture the graph
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.stream):
                    h0 = self.h0.to(x.device, non_blocking=True).contiguous()
                    c0 = self.c0.to(x.device, non_blocking=True).contiguous()
                    _, (_, static_output) = self.lstm(static_input, (h0, c0))
            
            return static_input, static_output, graph
            
        except Exception:
            # If graph capture fails, return None
            return None, None, None
    
    def forward(self, x):
        """
        Forward pass through the LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :return: The output tensor, shape (num_layers, batch_size, hidden_size)
        """
        # Ultra-fast path for the most recent input pattern
        if torch.cuda.is_available() and x.is_cuda:
            key = self._get_cache_key(x)
            
            # Check if this is the same as our most recent input
            if key == self.last_key and self.last_graph is not None:
                with torch.cuda.stream(self.stream):
                    self.last_static_input.copy_(x)
                    self.last_graph.replay()
                return self.last_static_output
            
            # Ensure input is contiguous for optimal memory access
            if not x.is_contiguous():
                x = x.contiguous()
            
            # Check if we have this in our main cache
            if key in self.graph_cache:
                # Move to most recently used position
                static_input, static_output, graph = self.graph_cache.pop(key)
                self.graph_cache[key] = (static_input, static_output, graph)
                
                # Update fast path cache
                self.last_key = key
                self.last_static_input = static_input
                self.last_static_output = static_output
                self.last_graph = graph
                
                # Execute the graph
                with torch.cuda.stream(self.stream):
                    static_input.copy_(x)
                    graph.replay()
                return static_output
            
            # Create a new graph
            static_input, static_output, graph = self._create_cuda_graph(x)
            
            if static_input is not None:
                # Update fast path cache
                self.last_key = key
                self.last_static_input = static_input
                self.last_static_output = static_output
                self.last_graph = graph
                
                # Add to main cache
                self.graph_cache[key] = (static_input, static_output, graph)
                
                # Maintain cache size
                while len(self.graph_cache) > self.max_cache_size:
                    self.graph_cache.popitem(last=False)
                
                # Execute the graph
                with torch.cuda.stream(self.stream):
                    static_input.copy_(x)
                    graph.replay()
                return static_output
        
        # Fallback path for CPU tensors or graph creation failures
        # Ensure hidden states are on the correct device
        h0 = self.h0.to(x.device)
        c0 = self.c0.to(x.device)  # Fixed: correctly use c0 instead of h0
        
        # Try scripted LSTM first if available
        if self.scripted_lstm is not None:
            try:
                _, (_, c_n) = self.scripted_lstm(x, (h0, c0))
                return c_n
            except Exception:
                pass
        
        # Fall back to regular LSTM
        _, (_, c_n) = self.lstm(x, (h0, c0))
        return c_n

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
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