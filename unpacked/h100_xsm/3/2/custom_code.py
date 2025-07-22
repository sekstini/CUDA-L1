import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        # Create weight and bias parameters directly
        self.w1 = nn.Parameter(torch.empty(hidden_layer_sizes[0], input_size))
        self.b1 = nn.Parameter(torch.empty(hidden_layer_sizes[0]))
        
        self.w2 = nn.Parameter(torch.empty(hidden_layer_sizes[1], hidden_layer_sizes[0]))
        self.b2 = nn.Parameter(torch.empty(hidden_layer_sizes[1]))
        
        self.w3 = nn.Parameter(torch.empty(output_size, hidden_layer_sizes[1]))
        self.b3 = nn.Parameter(torch.empty(output_size))
        
        # Pre-transposed weights for faster computation during forward pass
        self.w1_t = None
        self.w2_t = None
        self.w3_t = None
        
        # Initialize weights and biases
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Standard initialization for linear layers (same as nn.Linear)
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w1)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b1, -bound, bound)
        
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w2)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b2, -bound, bound)
        
        nn.init.kaiming_uniform_(self.w3, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w3)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b3, -bound, bound)
    
    def _update_transposed_weights(self, device):
        # Store transposed weights to avoid transposition during forward pass
        self.w1_t = self.w1.t().contiguous().to(device)
        self.w2_t = self.w2.t().contiguous().to(device)
        self.w3_t = self.w3.t().contiguous().to(device)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Ensure input is contiguous for optimal memory access
        x = x.contiguous()
        
        # Update transposed weights if needed
        if self.w1_t is None or self.w1_t.device != x.device:
            self._update_transposed_weights(x.device)
        
        # First layer: Linear + ReLU
        # Using optimized torch.addmm for fused matrix multiplication and bias addition
        h1 = torch.addmm(self.b1, x, self.w1_t)
        h1 = F.relu(h1, inplace=True)  # In-place ReLU to save memory
        
        # Second layer: Linear + ReLU
        h2 = torch.addmm(self.b2, h1, self.w2_t)
        h2 = F.relu(h2, inplace=True)  # In-place ReLU to save memory
        
        # Output layer: Linear only
        out = torch.addmm(self.b3, h2, self.w3_t)
        
        return out

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 1
input_size = 1000
hidden_layer_sizes = [2000, 2000]  # Example of deep and narrow layers
output_size = 10

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]