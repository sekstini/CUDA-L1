import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelShuffleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, groups):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // groups
        
        # Save for backward
        ctx.groups = groups
        
        # Create indices for the shuffled channels - only compute once per shape/device
        if not hasattr(ChannelShuffleFunction, 'indices_cache'):
            ChannelShuffleFunction.indices_cache = {}
        
        cache_key = (channels, groups, x.device)
        if cache_key not in ChannelShuffleFunction.indices_cache:
            # Calculate indices for the channel shuffle
            indices = torch.arange(channels, device=x.device)
            indices = indices.view(groups, channels_per_group).t().contiguous().view(-1)
            ChannelShuffleFunction.indices_cache[cache_key] = indices
        
        indices = ChannelShuffleFunction.indices_cache[cache_key]
        
        # Use index_select for efficient shuffling
        output = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        output = output.index_select(-1, indices)
        output = output.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        groups = ctx.groups
        batch_size, channels, height, width = grad_output.size()
        channels_per_group = channels // groups
        
        # Get inverse indices for backward pass
        cache_key = (channels, groups, grad_output.device)
        if not hasattr(ChannelShuffleFunction, 'inverse_indices_cache'):
            ChannelShuffleFunction.inverse_indices_cache = {}
        
        if cache_key not in ChannelShuffleFunction.inverse_indices_cache:
            # Calculate indices for the inverse shuffle
            indices = torch.arange(channels, device=grad_output.device)
            indices = indices.view(channels_per_group, groups).t().contiguous().view(-1)
            ChannelShuffleFunction.inverse_indices_cache[cache_key] = indices
        
        inverse_indices = ChannelShuffleFunction.inverse_indices_cache[cache_key]
        
        # Use index_select for efficient inverse shuffling
        grad_input = grad_output.permute(0, 2, 3, 1)  # [B, H, W, C]
        grad_input = grad_input.index_select(-1, inverse_indices)
        grad_input = grad_input.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return grad_input, None

class OptimizedChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(OptimizedChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        return ChannelShuffleFunction.apply(x, self.groups)

class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnit, self).__init__()
        
        # Ensure the output channels are divisible by groups
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        # First 1x1 group convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Depthwise 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Second 1x1 group convolution
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Optimized shuffle operation
        self.shuffle = OptimizedChannelShuffle(groups)
        
        # Shortcut connection if input and output channels are the same
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = F.relu(self.bn3(self.conv3(out)))
        
        out += self.shortcut(x)
        return out

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super(ModelNew, self).__init__()
        
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        
        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.fc = nn.Linear(1024, num_classes)
    
    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = []
        layers.append(ShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes]