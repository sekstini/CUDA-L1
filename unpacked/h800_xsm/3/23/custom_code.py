import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(OptimizedMBConvBlock, self).__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = round(in_channels * expand_ratio)
        
        # Expansion phase
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.expand_bn = nn.BatchNorm2d(hidden_dim)
        
        # Depthwise phase
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                                       padding=1, groups=hidden_dim, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        
        # Projection phase
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        identity = x if self.use_residual else None
        
        # Expansion
        x = self.expand_conv(x)
        x = self.expand_bn(x)
        x = F.relu6(x, inplace=True)
        
        # Depthwise
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = F.relu6(x, inplace=True)
        
        # Projection
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
            
        return x

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB1 architecture implementation.

        :param num_classes: The number of output classes (default is 1000 for ImageNet).
        """
        super(ModelNew, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks
        self.mbconv1 = OptimizedMBConvBlock(32, 16, 1, 1)
        self.mbconv2 = OptimizedMBConvBlock(16, 24, 2, 6)
        self.mbconv3 = OptimizedMBConvBlock(24, 40, 2, 6)
        self.mbconv4 = OptimizedMBConvBlock(40, 80, 2, 6)
        self.mbconv5 = OptimizedMBConvBlock(80, 112, 1, 6)
        self.mbconv6 = OptimizedMBConvBlock(112, 192, 2, 6)
        self.mbconv7 = OptimizedMBConvBlock(192, 320, 1, 6)
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
        
        # For CUDA graph optimization
        self.static_input = None
        self.static_output = None
        self.graph = None
        self.warmed_up = False
        self.stream = None
        
        # For memory format optimization
        self.channels_last = False
        
        # Set model to evaluation mode for inference optimizations
        self.eval()
        
        # Convert to channels_last memory format immediately if on CUDA
        if torch.cuda.is_available():
            self.to(memory_format=torch.channels_last)
            self.channels_last = True
            self.stream = torch.cuda.Stream(priority=-1)  # High priority stream
    
    def _ensure_channels_last(self, x):
        """Ensure input tensor is in channels_last memory format if on CUDA"""
        if x.is_cuda and self.channels_last and not x.is_contiguous(memory_format=torch.channels_last):
            return x.contiguous(memory_format=torch.channels_last)
        return x
    
    def _forward_impl(self, x):
        """Implementation of the forward pass"""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        # MBConv blocks
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        
        # Final stages
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def forward(self, x):
        """
        Forward pass of the EfficientNetB1 model.

        :param x: Input tensor, shape (batch_size, 3, 240, 240)
        :return: Output tensor, shape (batch_size, num_classes)
        """
        # Ensure input is in channels_last format if on CUDA
        x = self._ensure_channels_last(x)
        
        # Use CUDA graph for optimization if on GPU
        if x.is_cuda and torch.cuda.is_available():
            # Initialize or update static tensors if input shape changes
            if (self.static_input is None or 
                self.static_input.shape != x.shape or 
                self.static_input.device != x.device):
                
                # Clean up previous graph and tensors if they exist
                if self.graph is not None:
                    del self.graph
                    self.graph = None
                
                self.static_input = torch.zeros_like(x, memory_format=torch.channels_last)
                self.static_output = None
                self.warmed_up = False
                
                if self.stream is None:
                    self.stream = torch.cuda.Stream(priority=-1)
            
            # Create and capture CUDA graph if needed
            if not self.warmed_up:
                try:
                    # Ensure model is in eval mode
                    self.eval()
                    
                    # Warm up with multiple iterations
                    with torch.cuda.stream(self.stream):
                        with torch.inference_mode(), torch.no_grad():
                            for _ in range(30):  # Optimal warmup iterations
                                self._forward_impl(x)
                        
                        # Capture graph
                        self.static_input.copy_(x)
                        self.graph = torch.cuda.CUDAGraph()
                        
                        with torch.cuda.graph(self.graph, stream=self.stream):
                            self.static_output = self._forward_impl(self.static_input)
                    
                    self.warmed_up = True
                except Exception:
                    # Fallback if CUDA graph capture fails
                    self.warmed_up = False
                    self.graph = None
            
            # Execute the captured graph if available
            if self.warmed_up and self.graph is not None:
                try:
                    with torch.cuda.stream(self.stream):
                        self.static_input.copy_(x)
                        self.graph.replay()
                        return self.static_output
                except Exception:
                    # Fallback if graph replay fails
                    with torch.inference_mode():
                        return self._forward_impl(x)
        
        # Standard forward pass if not using CUDA graph
        with torch.inference_mode():
            return self._forward_impl(x)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
input_shape = (3, 240, 240)
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [num_classes]