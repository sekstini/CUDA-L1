import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        :param downsample: Downsample layer for the shortcut connection
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: Number of output classes
        """
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # CUDA graph optimization resources
        self.static_input = None
        self.static_output = None
        self.graph = None
        self.stream = None
        self.warmup_done = False
        self.input_shape = None
        self.graph_ready = False
        self.device = None
        self.fallback_mode = False
        self.graph_capture_attempts = 0
        self.max_capture_attempts = 2
        self.use_cuda_graph = torch.cuda.is_available()
        
        # Enable memory pools if available (PyTorch 1.11+)
        if hasattr(torch.cuda, 'memory_stats') and torch.cuda.is_available():
            try:
                torch.cuda.memory.set_per_process_memory_fraction(0.95)
                if hasattr(torch.cuda, 'memory_stats'):
                    torch.cuda.memory_stats(self.device)
            except:
                pass
        
        # Initialize model in eval mode for inference optimizations
        self.eval()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def _forward_impl(self, x):
        """
        Implementation of the forward pass
        
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _clean_cuda_resources(self):
        """Clean up CUDA resources for reinitialization"""
        if hasattr(self, 'graph') and self.graph is not None:
            del self.graph
            self.graph = None
            
        if hasattr(self, 'static_input') and self.static_input is not None:
            del self.static_input
            self.static_input = None
            
        if hasattr(self, 'static_output') and self.static_output is not None:
            del self.static_output
            self.static_output = None
            
        # Clear CUDA cache to avoid memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _initialize_cuda_graph(self, x):
        """Initialize or reinitialize CUDA graph with proper error handling"""
        # Increment capture attempt counter
        self.graph_capture_attempts += 1
        
        # If we've tried too many times, fall back to standard execution
        if self.graph_capture_attempts > self.max_capture_attempts:
            self.fallback_mode = True
            return False
            
        # Clean up previous resources
        self._clean_cuda_resources()
            
        try:
            # Initialize static tensors on the same device as x
            self.static_input = torch.zeros_like(x, device=x.device)
            self.input_shape = x.shape
            self.device = x.device
            
            # Create new graph
            self.graph = torch.cuda.CUDAGraph()
            
            # Warmup pass to ensure all operations are initialized
            with torch.no_grad():
                _ = self._forward_impl(x)
                torch.cuda.synchronize()
            
            # Copy input data to static input tensor
            self.static_input.copy_(x)
            
            # Capture the graph
            with torch.cuda.graph(self.graph):
                self.static_output = self._forward_impl(self.static_input)
                
            # Quick validation by replaying the graph
            self.graph.replay()
            
            self.graph_ready = True
            return True
        except Exception:
            # Clean up resources on failure
            self._clean_cuda_resources()
            return False

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, 3, height, width)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        # Check if CUDA is available and we should use CUDA graph
        if not self.use_cuda_graph:
            with torch.no_grad():
                return self._forward_impl(x)
        
        # If in fallback mode, just use standard execution
        if self.fallback_mode:
            with torch.no_grad():
                return self._forward_impl(x)
        
        # Initialize device if not already done
        if self.device is None:
            self.device = next(self.parameters()).device
            
        # Initialize stream if not already done
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=self.device)
            
        # If input is not on the same device as model, move it
        if x.device != self.device:
            x = x.to(self.device)
        
        # Perform warmup if not done yet
        if not self.warmup_done:
            self._warmup(x)
            
        # Use CUDA graph for inference if available
        if not self.training:
            # Check if we need to initialize or update CUDA graph due to shape change
            if self.static_input is None or self.graph is None or x.shape != self.input_shape or not self.graph_ready:
                with torch.cuda.stream(self.stream):
                    success = self._initialize_cuda_graph(x)
                    if not success and not self.fallback_mode:
                        # If graph initialization failed but we're not in fallback mode yet,
                        # try one more time with a clean CUDA cache
                        torch.cuda.empty_cache()
                        success = self._initialize_cuda_graph(x)
                    
            # Use graph if ready, otherwise fall back to regular execution
            if self.graph_ready:
                # Copy input data to static input tensor and replay the graph
                # Use a single with-block to minimize overhead
                with torch.cuda.stream(self.stream):
                    self.static_input.copy_(x)
                    self.graph.replay()
                    # Return the result directly without cloning to avoid extra memory allocation
                    return self.static_output
            
        # Standard forward pass when graph is not available or not ready
        with torch.no_grad():
            return self._forward_impl(x)
    
    def _warmup(self, x):
        """Perform progressive warmup passes to initialize optimizations"""
        if not self.use_cuda_graph or self.warmup_done:
            return
            
        with torch.cuda.stream(self.stream):
            with torch.no_grad():
                # First pass with synchronization
                _ = self._forward_impl(x)
                torch.cuda.synchronize()
                
                # Multiple passes with increasing complexity and minimal synchronization
                for i in range(3):
                    _ = self._forward_impl(x)
                    if i == 1:  # Synchronize only once in the middle to reduce overhead
                        torch.cuda.synchronize()
                
                # Final synchronization to ensure all operations are complete
                torch.cuda.synchronize()
            
        self.warmup_done = True
    
    def __del__(self):
        """Clean up CUDA resources"""
        if hasattr(self, 'stream') and self.stream is not None:
            try:
                self.stream.synchronize()
            except:
                pass
            
        self._clean_cuda_resources()

# Keep all hyperparameters exactly as in the reference implementation
batch_size = 2
num_classes = 1000
input_shape = (batch_size, 3, 224, 224)

def get_inputs():
    return [torch.randn(input_shape)]

def get_init_inputs():
    return [num_classes]