import torch 
import torch.nn as nn 
from .base_network import BaseNetwork
from .res_block import BasicBlock
from typing import Type, Union, List, Optional
from pathlib import Path
import sys
# Calculate the path to the root of the project
# This assumes that `python_files` is a subdirectory in your project's root
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from enums.modelArchEnum import ModelType
#18 layers, one convolutional layer to start (conv1, bn1), 4 layers inbetween, each w/ 2 blocks. Each block
# has 2 conv layers (conv, bn). Therefore, 2 blocks * 2 layers = 4 conv layers. 4 conv layers * 4 layers = 16 conv layers 
# Last layer is fully-connected layer. therefore 1 initial conv layer + 16 inbetween conv layers + 1 fc layer = 18 layers
class ResNet18(BaseNetwork):
    #Init using the residual block class import 
    def __init__(self, block : Type[BasicBlock],  num_blocks_list: List[int],base_channels: int = 9, input_channels: int = 1, kernel_size_conv: int = 3,
                  stride_conv :int = 1, padding_conv: int = 1, bias: bool = False, output_size:int = 10) -> None:
        
        super(ResNet18, self).__init__()

        #Initial convolutional and batch norm layers 
        self.conv1 = nn.Conv2d(input_channels, base_channels , kernel_size_conv, stride= stride_conv,
                                padding = padding_conv, bias = bias)
        
        self.bn1 = nn.BatchNorm2d(base_channels)
        #Sets the number of output channels from the initial convolution for calculating input/output size in future layers 
        self.in_channels = base_channels
        #Four inbetween layers
        self.layer1: nn.Sequential = self._make_layer(block, num_blocks_list[0], base_channels, stride = 1, kernel_size = 3)
        self.layer2: nn.Sequential = self._make_layer(block, num_blocks_list[1], base_channels * 2, stride = 1, kernel_size = 3)
        self.layer3: nn.Sequential = self._make_layer(block, num_blocks_list[2], base_channels * 4, stride = 1, kernel_size = 3)
        self.layer4: nn.Sequential = self._make_layer(block, num_blocks_list[3], base_channels * 8, stride = 1, kernel_size = 3)
        #Final fully-connected layer 
        self.fc:nn.Linear = nn.Linear(base_channels * 8 * block.expansion, output_size )
        # Rectified linear 
        self.relu:nn.ReLU = nn.ReLU(inplace = True)
        #Max pooling 
        self.max_pool: nn.MaxPool2d = nn.MaxPool2d(kernel_size = kernel_size_conv, stride = stride_conv, padding = padding_conv)
        #Avg pooling 
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        dummy_block: block = block(self.in_channels,output_size, conv_stride = 1, kernel_size = 3)

        self.hyperparams: dict = {
            "res_block": dummy_block.get_resblock(), 
            "num_blocks_list": num_blocks_list, 
            "base_channels": base_channels, 
            "input_channels": input_channels, 
            "kernel_size_conv": kernel_size_conv, 
            "stride_conv": stride_conv, 
            "padding_conv": padding_conv, 
            "bias": bias, 
            "output_size": output_size,
        }

    #Create an inbetween layer using the residual blocks 
    def _make_layer(self, block:Type[BasicBlock], num_blocks:int, output_size:int,
                    stride:int, kernel_size:int) -> nn.Sequential:
        #List of layers 
        layers: List[nn.Module] = []
        #stride array
        strides: List[int] = [stride] + [1] * (num_blocks-1)
        #Add layers to the layers list 
        for stride in strides:
            layers.append(block(self.in_channels,output_size, stride, kernel_size))
            #Change the input size of this layer based to the output size of the previous layer 
            self.in_channels = output_size * block.expansion
        # *layers is python's spread operator (...) in JS, spreads the array
        return nn.Sequential(*layers)  
    # Forward training 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)
        #Run through layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #pool
        out = self.avg_pool(out)
        #Flatten into a 2d tensor for the fully-connected layer 
        out = torch.flatten(out, 1)
        #Run through fully-connected layer 
        out = self.fc(out)

        return out
     
    # Return model name 
    def get_model_type(self) -> ModelType:
        return ModelType.RESNET18

    def get_hyperparams(self) -> dict:
        return self.hyperparams

# Function to return a resnet model with 18 layers 
def create_ResNet18(input_channels:int = 1, output_size:int = 10):
    return ResNet18(BasicBlock, [2,2,2,2], input_channels = input_channels, output_size = output_size)