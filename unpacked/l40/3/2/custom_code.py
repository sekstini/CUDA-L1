import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        # Create weights and biases directly as parameters
        # Store weights pre-transposed for torch.addmm which expects (in_features, out_features)
        self.weight1 = nn.Parameter(torch.Tensor(hidden_layer_sizes[0], input_size))
        self.bias1 = nn.Parameter(torch.Tensor(hidden_layer_sizes[0]))
        
        self.weight2 = nn.Parameter(torch.Tensor(hidden_layer_sizes[1], hidden_layer_sizes[0]))
        self.bias2 = nn.Parameter(torch.Tensor(hidden_layer_sizes[1]))
        
        self.weight3 = nn.Parameter(torch.Tensor(output_size, hidden_layer_sizes[1]))
        self.bias3 = nn.Parameter(torch.Tensor(output_size))
        
        # Initialize parameters
        self.reset_parameters()
        
        # Pre-transpose weights for more efficient matrix multiplication
        self.weight1_t = nn.Parameter(self.weight1.t(), requires_grad=False)
        self.weight2_t = nn.Parameter(self.weight2.t(), requires_grad=False)
        self.weight3_t = nn.Parameter(self.weight3.t(), requires_grad=False)
    
    def reset_parameters(self):
        # Initialize weights using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight1)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias1, -bound, bound)
        
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight2)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias2, -bound, bound)
        
        nn.init.kaiming_uniform_(self.weight3, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight3)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias3, -bound, bound)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Ensure input is contiguous for better memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # First hidden layer with ReLU
        hidden1 = torch.addmm(self.bias1, x, self.weight1_t)
        hidden1.relu_()  # In-place ReLU to save memory
        
        # Second hidden layer with ReLU
        hidden2 = torch.addmm(self.bias2, hidden1, self.weight2_t)
        hidden2.relu_()  # In-place ReLU to save memory
        
        # Output layer (linear only)
        output = torch.addmm(self.bias3, hidden2, self.weight3_t)
        
        return output

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 1
input_size = 1000
hidden_layer_sizes = [2000, 2000]  # Example of deep and narrow layers
output_size = 10

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [input_size, hidden_layer_sizes, output_size]