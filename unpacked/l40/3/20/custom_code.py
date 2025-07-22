import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    """Optimized Conv-BatchNorm-ReLU6 block with fusion capabilities"""
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6(inplace=True)
        self.fused = False
    
    def forward(self, x):
        if self.fused:
            return self.relu(self.conv(x))
        else:
            return self.relu(self.bn(self.conv(x)))
    
    def fuse_bn(self):
        """Fuse batch norm into conv for inference efficiency"""
        if self.fused:
            return
            
        w = self.conv.weight
        mean = self.bn.running_mean
        var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps
        
        std = torch.sqrt(var + eps)
        t = gamma / std
        
        self.conv.weight.data = w * t.view(-1, 1, 1, 1)
        
        if self.conv.bias is None:
            self.conv.bias = nn.Parameter(torch.zeros_like(mean))
        
        self.conv.bias.data = beta - mean * t
        self.fused = True

class ConvBN(nn.Module):
    """Optimized Conv-BatchNorm block with fusion capabilities (no ReLU)"""
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBN, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.fused = False
    
    def forward(self, x):
        if self.fused:
            return self.conv(x)
        else:
            return self.bn(self.conv(x))
    
    def fuse_bn(self):
        """Fuse batch norm into conv for inference efficiency"""
        if self.fused:
            return
            
        w = self.conv.weight
        mean = self.bn.running_mean
        var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps
        
        std = torch.sqrt(var + eps)
        t = gamma / std
        
        self.conv.weight.data = w * t.view(-1, 1, 1, 1)
        
        if self.conv.bias is None:
            self.conv.bias = nn.Parameter(torch.zeros_like(mean))
        
        self.conv.bias.data = beta - mean * t
        self.fused = True

class InvertedResidual(nn.Module):
    """Optimized Inverted Residual Block for MobileNetV2"""
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        
        layers = []
        if expand_ratio != 1:
            # Pointwise convolution
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        
        # Depthwise convolution
        layers.append(ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim))
        
        # Pointwise linear convolution
        layers.append(ConvBN(hidden_dim, oup, kernel_size=1))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
    
    def fuse_bn(self):
        """Fuse all batch norms in this block"""
        for module in self.conv:
            if hasattr(module, 'fuse_bn'):
                module.fuse_bn()

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        MobileNetV2 architecture implementation in PyTorch.

        :param num_classes: The number of output classes. Default is 1000.
        """
        super(ModelNew, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            """
            This function ensures that the number of channels is divisible by the divisor.
            """
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        # MobileNetV2 architecture
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer
        self.features = nn.ModuleList([ConvBNReLU(3, input_channel, stride=2)])

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Building last several layers
        self.features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        # Linear layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
        # CUDA graph related attributes
        self._cuda_graph_captured = False
        self._static_input = None
        self._static_output = None
        self._graph = None
        self._stream = None
        self._warmup_iterations = 10  # Optimal warmup iterations from best implementation
        
        # Set model to evaluation mode and optimize
        self.eval()
        self._optimize_for_inference()
    
    def _optimize_for_inference(self):
        """Apply inference-time optimizations"""
        # Fuse batch norms for inference efficiency
        for module in self.features:
            if hasattr(module, 'fuse_bn'):
                module.fuse_bn()
        
        # Convert to channels_last memory format for better performance
        self = self.to(memory_format=torch.channels_last)
        
        # Apply TorchScript to classifier for better performance
        try:
            self.classifier = torch.jit.script(self.classifier)
        except Exception:
            pass
        
        # GPU-specific optimizations
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.to(device)
            
            # Create high-priority CUDA stream for better performance
            self._stream = torch.cuda.Stream(priority=-1)
            
            with torch.inference_mode():
                dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
                dummy_input = dummy_input.to(memory_format=torch.channels_last)
                
                # Warmup with dedicated stream
                with torch.cuda.stream(self._stream):
                    for _ in range(self._warmup_iterations):
                        _ = self._forward_no_graph(dummy_input)
                
                # Ensure all operations are completed
                self._stream.synchronize()
                torch.cuda.synchronize()

    def _maybe_capture_cuda_graph(self, x):
        """Capture CUDA graph if not already captured"""
        if not torch.cuda.is_available() or self._cuda_graph_captured:
            return False
        
        if x.shape[0] != batch_size:
            return False
        
        try:
            # Create static tensors with optimal memory layout
            self._static_input = x.clone().detach()
            if not self._static_input.is_contiguous(memory_format=torch.channels_last):
                self._static_input = self._static_input.contiguous(memory_format=torch.channels_last)
            
            self._static_output = torch.empty(batch_size, num_classes, device=x.device)
            
            # Extended warmup before capture with synchronization
            torch.cuda.synchronize()
            with torch.cuda.stream(self._stream):
                for _ in range(self._warmup_iterations):
                    with torch.inference_mode():
                        _ = self._forward_no_graph(self._static_input)
            
            self._stream.synchronize()
            torch.cuda.synchronize()
            
            # Capture the graph
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(self._stream):
                with torch.cuda.graph(self._graph):
                    with torch.inference_mode():
                        self._static_output = self._forward_no_graph(self._static_input)
            
            self._cuda_graph_captured = True
            return True
        except Exception:
            # Reset if capture fails
            self._cuda_graph_captured = False
            self._static_input = None
            self._static_output = None
            self._graph = None
            return False

    def _forward_no_graph(self, x):
        """Forward pass without CUDA graph"""
        # Ensure channels_last format for better performance
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        
        # Process features efficiently
        for module in self.features:
            x = module(x)
        
        # Flatten and classify
        x = x.flatten(1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        """
        Forward pass of the MobileNetV2 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        with torch.inference_mode():
            # Try CUDA graph optimization if on GPU
            if torch.cuda.is_available() and x.is_cuda:
                if not self._cuda_graph_captured:
                    captured = self._maybe_capture_cuda_graph(x)
                    if not captured:
                        # Fall back to regular forward pass with stream
                        with torch.cuda.stream(self._stream):
                            return self._forward_no_graph(x)
                
                if self._cuda_graph_captured:
                    # Use CUDA graph for optimal performance
                    with torch.cuda.stream(self._stream):
                        self._static_input.copy_(x)
                        self._graph.replay()
                    return self._static_output
            
            # Fall back to regular forward pass
            return self._forward_no_graph(x)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]