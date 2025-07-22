import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Initialize the VGG16 model with optimizations.
        
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(ModelNew, self).__init__()
        
        # Enable cuDNN benchmarking for faster convolution algorithms
        cudnn.benchmark = True
        
        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        # Pooling layer (reused)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
        # Dropout layer (reused)
        self.dropout = nn.Dropout(p=0.0)
        
        # Convert model to channels_last memory format for better performance
        self = self.to(memory_format=torch.channels_last)
        
        # Create a dedicated CUDA stream for this model if CUDA is available
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda:
            self.stream = torch.cuda.Stream()
            
            # Perform a warmup pass to help cuDNN select optimal algorithms
            with torch.no_grad(), torch.cuda.stream(self.stream):
                dummy_input = torch.randn(batch_size, 3, 224, 224, device='cuda')
                dummy_input = dummy_input.to(memory_format=torch.channels_last)
                self.to('cuda')
                _ = self(dummy_input)
                torch.cuda.synchronize()
    
    def forward(self, x):
        """
        Forward pass of the VGG16 model with optimizations.
        
        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # Convert input to channels_last if not already in that format
        if self.has_cuda and not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        
        if self.has_cuda:
            with torch.cuda.stream(self.stream):
                # Block 1
                x = F.relu(self.conv1_1(x), inplace=True)
                x = F.relu(self.conv1_2(x), inplace=True)
                x = self.maxpool(x)
                
                # Block 2
                x = F.relu(self.conv2_1(x), inplace=True)
                x = F.relu(self.conv2_2(x), inplace=True)
                x = self.maxpool(x)
                
                # Block 3
                x = F.relu(self.conv3_1(x), inplace=True)
                x = F.relu(self.conv3_2(x), inplace=True)
                x = F.relu(self.conv3_3(x), inplace=True)
                x = self.maxpool(x)
                
                # Block 4
                x = F.relu(self.conv4_1(x), inplace=True)
                x = F.relu(self.conv4_2(x), inplace=True)
                x = F.relu(self.conv4_3(x), inplace=True)
                x = self.maxpool(x)
                
                # Block 5
                x = F.relu(self.conv5_1(x), inplace=True)
                x = F.relu(self.conv5_2(x), inplace=True)
                x = F.relu(self.conv5_3(x), inplace=True)
                x = self.maxpool(x)
                
                # Flatten and pass through classifier
                x = torch.flatten(x, 1)
                x = F.relu(self.fc1(x), inplace=True)
                x = self.dropout(x)
                x = F.relu(self.fc2(x), inplace=True)
                x = self.dropout(x)
                x = self.fc3(x)
                
                # Ensure computation is complete before returning
                torch.cuda.current_stream().wait_stream(self.stream)
        else:
            # Block 1
            x = F.relu(self.conv1_1(x), inplace=True)
            x = F.relu(self.conv1_2(x), inplace=True)
            x = self.maxpool(x)
            
            # Block 2
            x = F.relu(self.conv2_1(x), inplace=True)
            x = F.relu(self.conv2_2(x), inplace=True)
            x = self.maxpool(x)
            
            # Block 3
            x = F.relu(self.conv3_1(x), inplace=True)
            x = F.relu(self.conv3_2(x), inplace=True)
            x = F.relu(self.conv3_3(x), inplace=True)
            x = self.maxpool(x)
            
            # Block 4
            x = F.relu(self.conv4_1(x), inplace=True)
            x = F.relu(self.conv4_2(x), inplace=True)
            x = F.relu(self.conv4_3(x), inplace=True)
            x = self.maxpool(x)
            
            # Block 5
            x = F.relu(self.conv5_1(x), inplace=True)
            x = F.relu(self.conv5_2(x), inplace=True)
            x = F.relu(self.conv5_3(x), inplace=True)
            x = self.maxpool(x)
            
            # Flatten and pass through classifier
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x), inplace=True)
            x = self.dropout(x)
            x = F.relu(self.fc2(x), inplace=True)
            x = self.dropout(x)
            x = self.fc3(x)
        
        return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]