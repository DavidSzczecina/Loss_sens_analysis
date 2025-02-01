import torch
import torch.nn as nn 
from .base_network import BaseNetwork
from pathlib import Path
import sys
# Calculate the path to the root of the project
# This assumes that `python_files` is a subdirectory in your project's root
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from enums.modelArchEnum import ModelType
class SimpleMLP(BaseNetwork):
    def __init__(self, image_dimensions = 28, inputChannels = 1, hidden_size_1 = 256, hidden_size_2 = 128 ,output_size = 10, dropout = 0.2) -> None:
        super(SimpleMLP, self).__init__()
        #input_dimensions are the n x n size of the image, square to find teh total number of pixels
        self.image_size = image_dimensions**2
        self.image_dimensions = image_dimensions

        #Three layers, one input layer, one hidden, one output
        self.fc1: nn.Linear = nn.Linear(self.image_size*inputChannels, hidden_size_1)
        self.fc2: nn.Linear = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3: nn.Linear = nn.Linear(hidden_size_2, output_size)
    #    self.dropout = nn.Dropout(dropout)
        self.relu: nn.ReLU = nn.ReLU()

        self.hyperparams: dict = {
            "image_dimensions": self.image_dimensions, 
            "input_channels": inputChannels,
            "hidden_size_1": hidden_size_1, 
            "hidden_size_2": hidden_size_2, 
            "output_size": output_size, 
            "dropout": dropout
        }
    #Forward method, returning raw output logits 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
         # Flatten the input tensor to a vector
        x: torch.Tensor = x.view(x.size(0), -1)
        # Three fully-connected layers 
        out: torch.Tensor = self.fc1(x)
        out = self.relu(out)
     #   out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
     #   out = self.dropout(out)
        out = self.fc3(out)
        return out 

    #Return model name 
    def get_model_type(self) -> ModelType:
        return ModelType.MLP

    def get_hyperparams(self) -> dict:
        return self.hyperparams
