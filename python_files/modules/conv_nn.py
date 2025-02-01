import torch.nn as nn 
import torch.nn.functional as F 
import torch
from .base_network import BaseNetwork
from pathlib import Path
import sys
# Calculate the path to the root of the project
# This assumes that `python_files` is a subdirectory in your project's root
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from enums.modelArchEnum import ModelType
class ConvNN(BaseNetwork):
    def __init__(self, input_channels = 1, image_dimensions = 28, conv_size_1 =14,conv_size_2 = 28,fc1_size = 128, kernel_size_conv = 3, 
                 conv_stride = 1, pool_stride = 2,dropout_1 = 0.25, dropout_2 = 0.25 , kernel_size_pool = 2, output_size = 10) -> None:
        super(ConvNN, self).__init__()
        self.kernel_size_pool: int = kernel_size_pool
        #Convolutional layer 1, resulting output wil be of size 26x26 (28-3+1) given MNIST IS 28x28
        self.conv1: nn.Conv2d = nn.Conv2d(input_channels, conv_size_1, kernel_size_conv, conv_stride)
        
        out_size_1: int = image_dimensions- kernel_size_conv + 1
        #Convolutional layer 2, the output of layer 1 is the input of layer 2, resulting output will be of size 24x24 (26-3+1)
        self.conv2: nn.Conv2d = nn.Conv2d(conv_size_1, conv_size_2, kernel_size_conv, conv_stride)

        out_size_2: int = out_size_1 - kernel_size_conv + 1
         #Dropout layers, zeros a percentage of the weights
        self.dropout1: nn.Dropout = nn.Dropout(dropout_1)
        self.dropout2: nn.Dropout = nn.Dropout(dropout_2)

         #fully connected layers, in the forward function, maxpooling is used w/ a kernel of size 2x2
        # therefore, the 24x24 image after two convolution layers is reduced to 12x12. The flattening results in 
        # 12*12*output_channels_2 input values for the first fully connected layer
        fc1_factor: int = (out_size_2 // kernel_size_pool)**2

        #Fully connected layers 
        self.fc1: nn.Linear= nn.Linear(fc1_factor * conv_size_2, fc1_size)
        self.fc2: nn.Linear = nn.Linear(fc1_size, output_size)
        #Rectified Linear activation function
        self.relu: nn.ReLU = nn.ReLU()
        self.pool: nn.MaxPool2d = nn.MaxPool2d(kernel_size_pool, stride = pool_stride)

        self.hyperparams: dict = { 
            "input_channels": input_channels, 
            "image_dimensions": image_dimensions,
            "conv_size_1": conv_size_1, 
            "conv_size_2": conv_size_2, 
            "fc1_size": fc1_size, 
            "kernel_size_conv": kernel_size_conv, 
            "conv_stride": conv_stride, 
            "pool_stride": pool_stride, 
            "dropout_1": dropout_1, 
            "dropout_2": dropout_2, 
            "kernel_size_pool": kernel_size_pool, 
            "output_size": output_size,
        }

    # Forward method for training 
    def forward(self, x: torch.Tensor) ->  torch.Tensor:
        
        #Two convolutional layers followed by two fully-connected layers
        out: torch.Tensor = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.dropout1(out)

        out = torch.flatten(out, 1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        
        return out 
    
    #Return model name 
    def get_model_type(self) -> ModelType:
        return ModelType.CNN

    def get_hyperparams(self) -> dict:
        return self.hyperparams
    

