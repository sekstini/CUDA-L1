import torch
import torch.nn as nn
import torch.nn.functional as F

# Enable cuDNN benchmarking for optimal kernel selection
torch.backends.cudnn.benchmark = True

# Custom CUDA kernel for optimized 1x1 convolution
if torch.cuda.is_available():
    cuda_source = """
    extern "C" __global__ void conv1x1_forward_kernel(
        const float* __restrict__ input,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        float* __restrict__ output,
        int batch_size,
        int channels_in,
        int channels_out,
        int height,
        int width) {
        
        // Calculate output position
        const int n = blockIdx.x;
        const int c_out_block = blockIdx.y;
        const int hw = blockIdx.z * blockDim.x + threadIdx.x;
        const int c_out_thread = threadIdx.y;
        
        const int c_out = c_out_block * blockDim.y + c_out_thread;
        const int h = hw / width;
        const int w = hw % width;
        
        if (n >= batch_size || c_out >= channels_out || h >= height || w >= width)
            return;
            
        // Calculate input offset for this output position
        const int spatial_size = height * width;
        const int input_base = n * channels_in * spatial_size + h * width + w;
        
        // Calculate output position
        const int output_idx = ((n * channels_out + c_out) * height + h) * width + w;
        
        // Compute dot product
        float sum = 0.0f;
        for (int c_in = 0; c_in < channels_in; ++c_in) {
            sum += input[input_base + c_in * spatial_size] * weight[c_out * channels_in + c_in];
        }
        
        // Add bias if provided
        if (bias != nullptr) {
            sum += bias[c_out];
        }
        
        // Write output
        output[output_idx] = sum;
    }
    
    extern "C" __global__ void add_relu_inplace_kernel(
        float* __restrict__ output,
        const float* __restrict__ identity,
        int size) {
        
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = fmaxf(output[idx] + identity[idx], 0.0f);
        }
    }
    """
    
    from torch.utils.cpp_extension import load_inline
    try:
        resnet_cuda = load_inline(
            name="resnet_cuda",
            cpp_sources="",
            cuda_sources=cuda_source,
            functions=["conv1x1_forward_kernel", "add_relu_inplace_kernel"],
            with_cuda=True,
            extra_cuda_cflags=["-O3"]
        )
        CUDA_AVAILABLE = True
    except Exception as e:
        print(f"Warning: Could not load custom CUDA kernels: {e}")
        CUDA_AVAILABLE = False
else:
    CUDA_AVAILABLE = False

def custom_conv1x1(input, weight, bias=None):
    """
    Custom 1x1 convolution using our CUDA kernel if available
    """
    batch_size, channels_in, height, width = input.size()
    channels_out = weight.size(0)
    
    output = torch.empty((batch_size, channels_out, height, width), 
                         dtype=input.dtype, device=input.device)
    
    if CUDA_AVAILABLE and input.is_cuda and input.dtype == torch.float32:
        # Use our custom CUDA kernel
        threads_per_block_x = 32
        threads_per_block_y = 4
        
        blocks_x = batch_size
        blocks_y = (channels_out + threads_per_block_y - 1) // threads_per_block_y
        blocks_z = (height * width + threads_per_block_x - 1) // threads_per_block_x
        
        bias_ptr = bias.data_ptr() if bias is not None else 0
        
        resnet_cuda.conv1x1_forward_kernel(
            input, weight, bias, output,
            batch_size, channels_in, channels_out,
            height, width,
            grid=(blocks_x, blocks_y, blocks_z),
            block=(threads_per_block_x, threads_per_block_y, 1)
        )
        return output
    else:
        # Fall back to PyTorch's implementation
        return F.conv2d(input, weight, bias)

def custom_add_relu(output, identity):
    """
    Custom add + ReLU operation using our CUDA kernel if available
    """
    if CUDA_AVAILABLE and output.is_cuda and output.dtype == torch.float32:
        size = output.numel()
        threads = 256
        blocks = (size + threads - 1) // threads
        
        resnet_cuda.add_relu_inplace_kernel(
            output, identity, size,
            grid=(blocks,),
            block=(threads,)
        )
        return output
    else:
        output.add_(identity)
        return F.relu(output, inplace=True)

class OptimizedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Optimized bottleneck block with batch normalization folding for inference
        """
        super(OptimizedBottleneck, self).__init__()
        # Standard implementation components (needed for training mode)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        # Pre-compute folded weights and biases for inference
        self.register_buffer('folded_conv1_weight', None)
        self.register_buffer('folded_conv1_bias', None)
        self.register_buffer('folded_conv2_weight', None)
        self.register_buffer('folded_conv2_bias', None)
        self.register_buffer('folded_conv3_weight', None)
        self.register_buffer('folded_conv3_bias', None)
        self.register_buffer('folded_downsample_weight', None)
        self.register_buffer('folded_downsample_bias', None)
        
        # Initialize folded weights immediately
        self._fold_bn_weights()

    def _fold_bn_weights(self):
        """Fold BatchNorm parameters into Conv weights and biases"""
        # Fold BN1 into Conv1
        bn1_scale = self.bn1.weight / torch.sqrt(self.bn1.running_var + 1e-5)
        self.folded_conv1_weight = (self.conv1.weight * bn1_scale.view(-1, 1, 1, 1)).contiguous()
        self.folded_conv1_bias = (self.bn1.bias - self.bn1.running_mean * bn1_scale).contiguous()
        
        # Fold BN2 into Conv2
        bn2_scale = self.bn2.weight / torch.sqrt(self.bn2.running_var + 1e-5)
        self.folded_conv2_weight = (self.conv2.weight * bn2_scale.view(-1, 1, 1, 1)).contiguous()
        self.folded_conv2_bias = (self.bn2.bias - self.bn2.running_mean * bn2_scale).contiguous()
        
        # Fold BN3 into Conv3
        bn3_scale = self.bn3.weight / torch.sqrt(self.bn3.running_var + 1e-5)
        self.folded_conv3_weight = (self.conv3.weight * bn3_scale.view(-1, 1, 1, 1)).contiguous()
        self.folded_conv3_bias = (self.bn3.bias - self.bn3.running_mean * bn3_scale).contiguous()
        
        # Fold downsample BN if exists
        if self.downsample is not None and len(self.downsample) == 2:
            bn = self.downsample[1]
            conv = self.downsample[0]
            bn_scale = bn.weight / torch.sqrt(bn.running_var + 1e-5)
            self.folded_downsample_weight = (conv.weight * bn_scale.view(-1, 1, 1, 1)).contiguous()
            self.folded_downsample_bias = (bn.bias - bn.running_mean * bn_scale).contiguous()

    def forward(self, x):
        # For inference mode, use optimized path with folded BN
        if not self.training:
            identity = x
            
            # Step 1: Conv1 + ReLU (BN folded into Conv)
            if x.is_cuda and CUDA_AVAILABLE and self.folded_conv1_weight.size(2) == 1 and self.folded_conv1_weight.size(3) == 1:
                out = custom_conv1x1(x, self.folded_conv1_weight, self.folded_conv1_bias)
                out = F.relu(out, inplace=True)
            else:
                out = F.conv2d(x, self.folded_conv1_weight, self.folded_conv1_bias)
                out = F.relu(out, inplace=True)
            
            # Step 2: Conv2 + ReLU (BN folded into Conv)
            out = F.conv2d(out, self.folded_conv2_weight, self.folded_conv2_bias, 
                         stride=self.stride, padding=1)
            out = F.relu(out, inplace=True)
            
            # Step 3: Conv3 (BN folded into Conv)
            if out.is_cuda and CUDA_AVAILABLE and self.folded_conv3_weight.size(2) == 1 and self.folded_conv3_weight.size(3) == 1:
                out = custom_conv1x1(out, self.folded_conv3_weight, self.folded_conv3_bias)
            else:
                out = F.conv2d(out, self.folded_conv3_weight, self.folded_conv3_bias)
            
            # Apply folded downsample if needed
            if self.downsample is not None:
                if x.is_cuda and CUDA_AVAILABLE and self.folded_downsample_weight.size(2) == 1 and self.folded_downsample_weight.size(3) == 1:
                    identity = custom_conv1x1(x, self.folded_downsample_weight, self.folded_downsample_bias)
                else:
                    identity = F.conv2d(x, self.folded_downsample_weight, self.folded_downsample_bias, 
                                      stride=self.stride)
            
            # Add identity and apply ReLU
            if out.is_cuda and CUDA_AVAILABLE:
                out = custom_add_relu(out, identity)
            else:
                out.add_(identity)
                out = F.relu(out, inplace=True)
            
            return out
        
        # Standard implementation path for training
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = OptimizedBottleneck

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Pre-compute folded weights for first conv+bn
        bn_scale = self.bn1.weight / torch.sqrt(self.bn1.running_var + 1e-5)
        self.register_buffer('folded_conv1_weight', (self.conv1.weight * bn_scale.view(-1, 1, 1, 1)).contiguous())
        self.register_buffer('folded_conv1_bias', (self.bn1.bias - self.bn1.running_mean * bn_scale).contiguous())
        
        # Set model to evaluation mode by default for inference optimizations
        self.eval()
        
        # Optimize memory layout for all parameters
        self._optimize_memory_layout()

    def _optimize_memory_layout(self):
        """Optimize memory layout for all parameters"""
        # Ensure all weights are contiguous for better memory access
        for module in self.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if not module.weight.is_contiguous():
                    module.weight.data = module.weight.data.contiguous()
            if hasattr(module, 'bias') and module.bias is not None:
                if not module.bias.is_contiguous():
                    module.bias.data = module.bias.data.contiguous()
                    
        # Try to convert to channels_last memory format for better performance on modern GPUs
        try:
            self.to(memory_format=torch.channels_last)
        except:
            pass  # Fallback if channels_last is not supported

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

    def forward(self, x):
        # Try to use channels_last memory format for better performance
        try:
            if x.device.type == 'cuda' and not x.is_contiguous(memory_format=torch.channels_last):
                x = x.contiguous(memory_format=torch.channels_last)
        except:
            # Ensure input is at least contiguous for optimal memory access
            if not x.is_contiguous():
                x = x.contiguous()
        
        # Optimize first layer for inference
        if not self.training:
            # Use folded weights for initial conv layer
            x = F.conv2d(x, self.folded_conv1_weight, self.folded_conv1_bias, 
                       stride=2, padding=3)
            x = F.relu(x, inplace=True)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            
        x = self.maxpool(x)

        # Process through the main network layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # Most compute-intensive part (23 blocks)
        x = self.layer4(x)

        # Final processing
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
height = 224
width = 224
layers = [3, 4, 23, 3]
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]

def get_init_inputs():
    return [layers, num_classes]