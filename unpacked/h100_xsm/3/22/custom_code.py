import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        Optimized MBConv block implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(OptimizedMBConv, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.expand_ratio = expand_ratio
        hidden_dim = in_channels * expand_ratio
        
        # Create separate components for better optimization
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        else:
            self.expand = None
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, 
                     padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        """
        Forward pass of the optimized MBConv block.

        :param x: The input tensor
        :return: The output tensor
        """
        if self.use_residual:
            identity = x
        
        if self.expand is not None:
            x = self.expand(x)
        
        x = self.depthwise(x)
        x = self.project(x)
        
        if self.use_residual:
            x = x + identity
        
        return x

class PoolAndFlatten(nn.Module):
    """Custom module to combine pooling and flattening for better optimization"""
    def __init__(self):
        super(PoolAndFlatten, self).__init__()
    
    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB0 architecture implementation in PyTorch with optimizations.

        :param num_classes: The number of output classes (default is 1000 for ImageNet).
        """
        super(ModelNew, self).__init__()
        
        # Initial convolutional layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # MBConv blocks
        self.blocks = nn.ModuleList([
            # MBConv1 (32, 16, 1, 1)
            OptimizedMBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            # MBConv6 (16, 24, 2, 6)
            OptimizedMBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (24, 24, 1, 6)
            OptimizedMBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (24, 40, 2, 6)
            OptimizedMBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (40, 40, 1, 6)
            OptimizedMBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (40, 80, 2, 6)
            OptimizedMBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (80, 80, 1, 6)
            OptimizedMBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (80, 112, 1, 6)
            OptimizedMBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 112, 1, 6)
            OptimizedMBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 192, 2, 6)
            OptimizedMBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            OptimizedMBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            OptimizedMBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 320, 1, 6)
            OptimizedMBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        ])
        
        # Final convolutional layer
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True)
        )
        
        # Custom pooling and flattening layer
        self.pool_flatten = PoolAndFlatten()
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
        
        # For CUDA Graph optimization
        self.static_inputs = {}
        self.static_outputs = {}
        self.graphs = {}
        self.streams = {}
        self.scripted_model = None
        self.warmup_iterations = 50  # Extensive warmup for better stability
        self.use_cuda_graph = True
        
        # Apply optimization techniques
        self._optimize_model()
    
    def _optimize_model(self):
        """Apply optimization techniques to the model."""
        if torch.cuda.is_available():
            # Set model to eval mode for optimization
            self.eval()
            
            try:
                # Convert model to channels_last memory format
                if hasattr(torch, 'channels_last'):
                    self = self.to(memory_format=torch.channels_last)
                
                # Pre-compile kernels to avoid compilation during graph capture
                self._precompile_kernels()
                
                # Optimize stem
                self.stem = torch.jit.script(self.stem)
                
                # Optimize MBConv blocks
                for i, block in enumerate(self.blocks):
                    # Optimize components individually for better fusion
                    if block.expand is not None:
                        self.blocks[i].expand = torch.jit.script(block.expand)
                    self.blocks[i].depthwise = torch.jit.script(block.depthwise)
                    self.blocks[i].project = torch.jit.script(block.project)
                
                # Optimize head
                self.head = torch.jit.script(self.head)
                
                # Optimize pooling and flattening
                self.pool_flatten = torch.jit.script(self.pool_flatten)
                
                # Try to optimize the entire model with tracing
                try:
                    example_input = torch.rand(1, 3, 224, 224)
                    if hasattr(torch, 'channels_last'):
                        example_input = example_input.to(memory_format=torch.channels_last)
                    if torch.cuda.is_available():
                        example_input = example_input.cuda()
                    self.scripted_model = torch.jit.trace(self, example_input)
                    
                    # Warm up the scripted model
                    with torch.no_grad():
                        for _ in range(20):
                            self.scripted_model(example_input)
                except Exception:
                    self.scripted_model = None
                
                # Pre-warm the CUDA graph system for the expected batch size
                if torch.cuda.is_available() and self.use_cuda_graph:
                    try:
                        example_batch = torch.rand(batch_size, 3, 224, 224)
                        if hasattr(torch, 'channels_last'):
                            example_batch = example_batch.to(memory_format=torch.channels_last)
                        example_batch = example_batch.cuda()
                        self._warmup_cuda_graph(example_batch)
                    except Exception:
                        self.use_cuda_graph = False
                
            except Exception:
                # Fall back to original if optimization fails
                pass
    
    def _precompile_kernels(self):
        """Pre-compile CUDA kernels to avoid compilation during graph capture."""
        if not torch.cuda.is_available():
            return
        
        try:
            # Create example inputs of different sizes to pre-compile kernels for various shapes
            example_shapes = [(1, 3, 224, 224), (batch_size, 3, 224, 224)]
            
            for shape in example_shapes:
                x = torch.rand(*shape).cuda()
                if hasattr(torch, 'channels_last'):
                    x = x.to(memory_format=torch.channels_last)
                
                # Run a forward pass to compile kernels
                with torch.no_grad():
                    # Stem
                    x = self.stem(x)
                    
                    # MBConv blocks
                    for block in self.blocks:
                        if block.expand is not None:
                            x = block.expand(x)
                        x = block.depthwise(x)
                        x = block.project(x)
                    
                    # Head
                    x = self.head(x)
                    
                    # Pooling and classification
                    x = self.pool_flatten(x)
                    x = self.fc(x)
            
            # Force CUDA synchronization to ensure kernels are compiled
            torch.cuda.synchronize()
        except Exception:
            # Ignore errors during precompilation
            pass
    
    def _warmup_cuda_graph(self, x):
        """Warmup CUDA graph for faster subsequent executions."""
        if not torch.cuda.is_available() or self.training:
            return False
        
        # Get input shape as key for graph dictionary
        batch_size = x.shape[0]
        key = f"{batch_size}_{x.shape[2]}_{x.shape[3]}"
        
        # If we already have a graph for this input shape, no need to create another
        if key in self.graphs:
            return True
        
        try:
            # Create static input tensor for this input shape
            self.static_inputs[key] = x.clone()
            
            # Create a dedicated stream for this graph
            self.streams[key] = torch.cuda.Stream()
            
            # Run warmup iterations to stabilize execution
            with torch.no_grad():
                for _ in range(self.warmup_iterations):
                    self._forward_no_graph(self.static_inputs[key])
            
            # Force CUDA synchronization to ensure all operations are complete
            torch.cuda.synchronize()
            
            # Prepare for graph capture
            self.static_outputs[key] = torch.zeros_like(
                self._forward_no_graph(self.static_inputs[key]), 
                device=x.device
            )
            
            # Capture the graph using a dedicated stream for better isolation
            stream = self.streams[key]
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.static_outputs[key] = self._forward_no_graph(self.static_inputs[key])
                self.graphs[key] = graph
            
            # Wait for graph capture to complete
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            
            return True
        except Exception:
            # Fall back to regular execution if CUDA graph fails
            if key in self.static_inputs:
                del self.static_inputs[key]
            if key in self.static_outputs:
                del self.static_outputs[key]
            if key in self.graphs:
                del self.graphs[key]
            if key in self.streams:
                del self.streams[key]
            return False
    
    def _forward_no_graph(self, x):
        """Forward pass without CUDA graph optimization."""
        # Try to use the scripted model if available and in eval mode
        if not self.training and self.scripted_model is not None:
            try:
                return self.scripted_model(x)
            except Exception:
                # Fall back to regular execution if scripted model fails
                pass
        
        # Convert input to channels_last if on CUDA and supported
        if x.device.type == 'cuda' and x.dim() == 4 and hasattr(torch, 'channels_last'):
            x = x.contiguous(memory_format=torch.channels_last)
        
        # Process through the initial block
        x = self.stem(x)
        
        # Process through MBConv blocks
        for block in self.blocks:
            x = block(x)
        
        # Process through the final block
        x = self.head(x)
        
        # Global average pooling and classification using optimized module
        x = self.pool_flatten(x)
        x = self.fc(x)
        
        return x
    
    def forward(self, x):
        """
        Forward pass of the EfficientNetB0 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # Skip graph optimization during training
        if self.training or not self.use_cuda_graph:
            return self._forward_no_graph(x)
        
        # Get input shape as key for graph dictionary
        batch_size = x.shape[0]
        key = f"{batch_size}_{x.shape[2]}_{x.shape[3]}"
        
        # Use CUDA graph if available for this input shape
        if torch.cuda.is_available() and key in self.graphs:
            try:
                # Use the appropriate stream for this graph
                with torch.cuda.stream(self.streams[key]):
                    self.static_inputs[key].copy_(x)
                    self.graphs[key].replay()
                    # Ensure the output is ready before returning
                    result = self.static_outputs[key].clone()
                return result
            except Exception:
                # If graph replay fails, fall back to regular execution
                pass
        
        # Try to create a graph for this input shape if not done yet
        if torch.cuda.is_available() and key not in self.graphs:
            graph_created = self._warmup_cuda_graph(x)
            if graph_created and key in self.graphs:
                try:
                    # Use the appropriate stream for this graph
                    with torch.cuda.stream(self.streams[key]):
                        self.static_inputs[key].copy_(x)
                        self.graphs[key].replay()
                        # Ensure the output is ready before returning
                        result = self.static_outputs[key].clone()
                    return result
                except Exception:
                    # If graph replay fails, fall back to regular execution
                    pass
        
        # Fall back to regular forward pass
        return self._forward_no_graph(x)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]