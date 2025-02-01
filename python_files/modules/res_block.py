import torch 
import torch.nn as nn 

# Residual block for use in ResNet model architectures
class BasicBlock(nn.Module):
    #Factor by which to increase out channels
    expansion = 1
    #Initialize the block
    def __init__(self, input_size: int, output_size: int, conv_stride: int, kernel_size: int, kernel_size_shortcut: int = 1) -> None:
        
        super(BasicBlock, self).__init__()
        #Padding calc to ensure that the convolutional layers do not reduce the size of the input ie. 28x28 input stays 28x28 output
        padding = (kernel_size - 1) // 2 
        #Conv layer 1
        self.conv1 = nn.Conv2d(input_size, output_size, kernel_size, conv_stride, padding)
        #Batch norm 1
        self.bn1 = nn.BatchNorm2d(output_size)
        #Conv layer 2 
        self.conv2 = nn.Conv2d(output_size, output_size, kernel_size, conv_stride, padding)
        #Batch norm 2
        self.bn2 = nn.BatchNorm2d(output_size)

        #Initialize the skip connection
        self.shortcut = nn.Sequential()
        # Rectified linear 
        self.relu = nn.ReLU()

        # Checks the stride from convolution if the output is the same as the input size, ensured it stays the same 
        if conv_stride != 1 or input_size != self.expansion * output_size: 
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_size, self.expansion * output_size, kernel_size_shortcut, conv_stride, padding = 0), 
                nn.BatchNorm2d(self.expansion * output_size)
            )

        self.res_block: dict = {
            "input_size": input_size, 
            "output_size": output_size, 
            "conv_stride": conv_stride, 
            "kernel_size_shortcut": kernel_size_shortcut,
        }
    # Forward training method 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out)
        
        #This is the skip connection f(x) + x
        out += self.shortcut(x)

        out = self.relu(out)
        return out 
    
    def get_resblock(self) -> dict: 
        return self.res_block


        
