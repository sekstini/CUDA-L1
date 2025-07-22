import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedDenseLayer(nn.Module):
    def __init__(self, in_features, growth_rate):
        super(OptimizedDenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False)
        # No dropout since probability is 0.0 (no-op)
        
    def forward(self, x):
        out = self.bn(x)
        out = F.relu(out, inplace=True)
        return self.conv(out)

class EfficientDenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate):
        super(EfficientDenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        # Create layers with increasing input features
        for i in range(num_layers):
            layer_input_features = num_input_features + i * growth_rate
            self.layers.append(OptimizedDenseLayer(layer_input_features, growth_rate))
    
    def forward(self, x):
        features = x
        
        for layer in self.layers:
            new_features = layer(features)
            # Efficient concatenation - directly append new features
            features = torch.cat([features, new_features], 1)
        
        return features

class OptimizedTransitionLayer(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(OptimizedTransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.conv(x)
        return self.pool(x)

class ModelNew(nn.Module):
    def __init__(self, growth_rate=32, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # Initial convolution and pooling - use individual modules for better control
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # DenseNet201 configuration
        num_features = 64
        block_layers = [6, 12, 48, 32]  # DenseNet201 layers per block
        
        # Build dense blocks and transition layers
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        
        for i, num_layers in enumerate(block_layers):
            # Add dense block
            block = EfficientDenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate
            )
            self.dense_blocks.append(block)
            
            # Update number of features after dense block
            num_features = num_features + num_layers * growth_rate
            
            # Add transition layer except after the last block
            if i != len(block_layers) - 1:
                transition = OptimizedTransitionLayer(
                    num_input_features=num_features,
                    num_output_features=num_features // 2
                )
                self.transition_layers.append(transition)
                num_features = num_features // 2
        
        # Final batch norm and classifier
        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        # Initial feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        
        # Process through dense blocks and transition layers
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)
        
        # Final processing and classification
        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
num_classes = 10
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, num_classes]