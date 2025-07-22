import torch
import torch.nn as nn
import torch.cuda as cuda
import gc

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model with CUDA graph optimization.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        """
        super(ModelNew, self).__init__()
        
        # Register hidden states as buffers to ensure they're moved to the correct device
        self.register_buffer('h0', torch.randn((num_layers, batch_size, hidden_size)))
        self.register_buffer('c0', torch.randn((num_layers, batch_size, hidden_size)))
        
        # Use PyTorch's optimized LSTM implementation
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
        # CUDA graph optimization variables
        self.static_input = None
        self.static_output = None
        self.graph = None
        self.graph_ready = False
        self.input_shape = None
        self.input_device = None
        self.input_dtype = None
        self.warmup_done = False
        self.warmup_iterations = 42  # Optimized number of warmup iterations
        
        # Streams for parallel operations (created lazily)
        self.graph_capture_stream = None
        self.execution_stream = None
        self.warmup_stream = None
        self.replay_stream = None
        
        # Events for synchronization (created lazily)
        self.warmup_done_event = None
        self.graph_capture_done_event = None
        self.replay_done_event = None
        
        # Cache for different input shapes, devices, and dtypes
        self.graph_cache = {}
        self.max_cache_size = 10  # Limit cache size to avoid memory issues
        self.cache_usage_count = {}  # Track usage for LRU eviction
        self.cache_access_counter = 0
        
        # Flag to track if we're in training mode
        self.is_training = False
    
    def _initialize_cuda_resources(self):
        """Initialize CUDA streams and events if not already created"""
        if self.graph_capture_stream is None:
            self.graph_capture_stream = cuda.Stream()
        
        if self.execution_stream is None:
            self.execution_stream = cuda.Stream()
        
        if self.warmup_stream is None:
            self.warmup_stream = cuda.Stream()
        
        if self.replay_stream is None:
            # Use high priority stream for replay to minimize latency
            self.replay_stream = cuda.Stream(priority=-1)  # High priority
        
        if self.warmup_done_event is None:
            self.warmup_done_event = cuda.Event(enable_timing=False)
        
        if self.graph_capture_done_event is None:
            self.graph_capture_done_event = cuda.Event(enable_timing=False)
        
        if self.replay_done_event is None:
            self.replay_done_event = cuda.Event(enable_timing=False)
    
    def _evict_least_used_cache_entry(self):
        """Evict the least recently used cache entry if cache is full"""
        if len(self.graph_cache) >= self.max_cache_size:
            # Find the least recently used entry
            min_usage = float('inf')
            lru_key = None
            
            for key, count in self.cache_usage_count.items():
                if count < min_usage:
                    min_usage = count
                    lru_key = key
            
            if lru_key is not None:
                # Clean up resources
                cached_data = self.graph_cache[lru_key]
                del cached_data['graph']
                del cached_data['input']
                del cached_data['output']
                
                # Remove from cache
                del self.graph_cache[lru_key]
                del self.cache_usage_count[lru_key]
                
                # Force garbage collection to release CUDA memory
                gc.collect()
                torch.cuda.empty_cache()
    
    def _perform_warmup(self, x):
        """Perform warmup iterations to ensure JIT compilation is completed"""
        if not self.warmup_done:
            with torch.cuda.stream(self.warmup_stream):
                for _ in range(self.warmup_iterations):
                    warmup_out, _ = self.lstm(x, (self.h0, self.c0))
                    self.fc(warmup_out[:, -1, :].contiguous())
                
                # Record event to signal warmup completion
                self.warmup_done_event.record(self.warmup_stream)
            
            # Wait for warmup to complete
            self.warmup_done_event.synchronize()
            self.warmup_done = True
    
    def _capture_graph(self, x):
        """Capture the CUDA graph for the forward pass"""
        # Generate a unique key for the input shape, device, and dtype
        cache_key = f"{x.shape}_{x.device}_{x.dtype}"
        
        # Check if we have a cached graph for this configuration
        if cache_key in self.graph_cache:
            # Update usage count for LRU tracking
            self.cache_access_counter += 1
            self.cache_usage_count[cache_key] = self.cache_access_counter
            
            cached_data = self.graph_cache[cache_key]
            self.graph = cached_data['graph']
            self.static_input = cached_data['input']
            self.static_output = cached_data['output']
            self.input_shape = x.shape
            self.input_device = x.device
            self.input_dtype = x.dtype
            self.graph_ready = True
            return True
        
        try:
            # Check if we need to evict a cache entry
            self._evict_least_used_cache_entry()
            
            # Initialize CUDA resources
            self._initialize_cuda_resources()
            
            # Store the current input configuration
            self.input_shape = x.shape
            self.input_device = x.device
            self.input_dtype = x.dtype
            
            # Create static tensors for graph capture
            # Using contiguous tensors with same device and dtype
            self.static_input = x.clone().contiguous()
            
            # Perform warmup iterations
            self._perform_warmup(x)
            
            # First run to get output shape
            with torch.no_grad():
                out, _ = self.lstm(x, (self.h0, self.c0))
                result = self.fc(out[:, -1, :].contiguous())
                self.static_output = result.clone().contiguous()
            
            # Capture the graph in the dedicated capture stream
            with torch.cuda.stream(self.graph_capture_stream):
                self.graph = cuda.CUDAGraph()
                with cuda.graph(self.graph):
                    # Operations to capture in the graph
                    static_out, _ = self.lstm(self.static_input, (self.h0, self.c0))
                    static_last = static_out[:, -1, :].contiguous()
                    self.static_output.copy_(self.fc(static_last))
                
                # Record event to signal graph capture completion
                self.graph_capture_done_event.record(self.graph_capture_stream)
            
            # Wait for graph capture to complete
            self.graph_capture_done_event.synchronize()
            
            # Mark graph as ready for use
            self.graph_ready = True
            
            # Update usage count for LRU tracking
            self.cache_access_counter += 1
            self.cache_usage_count[cache_key] = self.cache_access_counter
            
            # Cache the graph for this input configuration
            self.graph_cache[cache_key] = {
                'graph': self.graph,
                'input': self.static_input,
                'output': self.static_output
            }
            
            return True
            
        except Exception:
            # If graph capture fails, disable graph usage for this configuration
            self.graph_ready = False
            
            # Clean up resources
            if self.graph is not None:
                del self.graph
                self.graph = None
            
            # Force garbage collection to release CUDA memory
            gc.collect()
            torch.cuda.empty_cache()
            
            return False
    
    def train(self, mode=True):
        """Override the train method to track training mode"""
        self.is_training = mode
        return super(ModelNew, self).train(mode)
    
    def eval(self):
        """Override the eval method to track training mode"""
        self.is_training = False
        return super(ModelNew, self).eval()
    
    def forward(self, x):
        """
        Forward pass through the LSTM model with CUDA graph optimization.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Skip graph optimization if in training mode
        if self.is_training:
            out, _ = self.lstm(x, (self.h0, self.c0))
            return self.fc(out[:, -1, :])
        
        # Fast path: use CUDA graph if available and input configuration matches
        if (x.is_cuda and 
            self.graph_ready and 
            x.shape == self.input_shape and
            x.device == self.input_device and
            x.dtype == self.input_dtype):
            
            # Initialize replay stream if not already done
            if self.replay_stream is None:
                self._initialize_cuda_resources()
            
            # Copy input data to static tensor and replay graph in dedicated stream
            with torch.cuda.stream(self.replay_stream):
                self.static_input.copy_(x, non_blocking=True)
                self.graph.replay()
                # Only record event if we'll need to synchronize
                if torch.is_grad_enabled():
                    self.replay_done_event.record(self.replay_stream)
            
            # Only synchronize if we need the result immediately
            if torch.is_grad_enabled():
                self.replay_done_event.synchronize()
                return self.static_output.clone()
            else:
                # Avoid unnecessary clone for inference
                return self.static_output
        
        # Standard execution path
        with torch.no_grad():  # Use no_grad for inference to reduce memory usage
            # Ensure input is contiguous for better memory access
            if not x.is_contiguous():
                x = x.contiguous()
            
            # Forward pass through LSTM and linear layer
            out, _ = self.lstm(x, (self.h0, self.c0))
            last_out = out[:, -1, :].contiguous()  # Get the last output and ensure contiguity
            result = self.fc(last_out)
            
            # Initialize CUDA graph on first CUDA input if not already done
            # or if input configuration has changed
            if x.is_cuda and (not self.graph_ready or 
                              x.shape != self.input_shape or 
                              x.device != self.input_device or
                              x.dtype != self.input_dtype):
                self._capture_graph(x)
            
            return result

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
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