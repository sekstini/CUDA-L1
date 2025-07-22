import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        :param num_layers: The number of layers in the dense block
        :param num_input_features: The number of input feature maps
        :param growth_rate: The growth rate for the dense block (new features added per layer)
        """
        super(ModelNew, self).__init__()
        self.num_layers = num_layers
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        self.final_features = num_input_features + num_layers * growth_rate
        
        # Create layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate
            self.layers.append(self._make_layer(in_features, growth_rate))
        
        # Register buffer for initialization tracking
        self.register_buffer('_initialized', torch.zeros(1))
        
    def _make_layer(self, in_features: int, growth_rate: int):
        """
        Creates a single layer with BatchNorm, ReLU, Conv2D, and Dropout.
        """
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )
    
    def _collect_layer_params(self):
        """
        Collect and cache layer parameters for faster access
        """
        self.weights = []
        self.bn_weights = []
        self.bn_biases = []
        self.running_means = []
        self.running_vars = []
        
        for layer in self.layers:
            bn = layer[0]
            conv = layer[2]
            
            self.weights.append(conv.weight)
            self.bn_weights.append(bn.weight)
            self.bn_biases.append(bn.bias)
            self.running_means.append(bn.running_mean)
            self.running_vars.append(bn.running_var)
    
    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Concatenated output tensor with shape (batch_size, num_output_features, height, width)
        """
        # Cache layer parameters on first run
        if self._initialized.item() == 0:
            self._collect_layer_params()
            self._initialized.fill_(1)
        
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
            
        batch_size, _, height, width = x.size()
        
        # Pre-allocate the output tensor with the final size
        output = torch.empty(batch_size, self.final_features, height, width, 
                          device=x.device, dtype=x.dtype)
        
        # Copy initial input to the output tensor
        output[:, :self.num_input_features].copy_(x)
        
        # Process each layer
        curr_features = self.num_input_features
        for i, layer in enumerate(self.layers):
            # Get current input - view of all features so far
            curr_input = output[:, :curr_features]
            
            # Apply the layer operations directly for better performance
            # 1. Batch Normalization
            if self.training:
                # In training mode, we need to calculate batch statistics
                bn = layer[0]
                curr_input = bn(curr_input)
            else:
                # In eval mode, use cached parameters
                normalized = F.batch_norm(
                    curr_input, 
                    self.running_means[i],
                    self.running_vars[i],
                    self.bn_weights[i],
                    self.bn_biases[i],
                    training=False,
                    momentum=0.1,
                    eps=1e-5
                )
                
                # 2. ReLU
                activated = F.relu(normalized)
                
                # 3. Convolution
                new_feature = F.conv2d(activated, self.weights[i], bias=None, stride=1, padding=1)
                
                # Copy the new feature to the output tensor
                output[:, curr_features:curr_features + self.growth_rate].copy_(new_feature)
            
            # Update the number of features
            curr_features += self.growth_rate
        
        return output

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
num_layers = 6
num_input_features = 32
growth_rate = 32
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, num_input_features, height, width)]

def get_init_inputs():
    return [num_layers, num_input_features, growth_rate]