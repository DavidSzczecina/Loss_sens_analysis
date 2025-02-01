import torch
import torch.nn as nn 
import torch.nn.functional as F 
import time
import os 
import sys
from pathlib import Path
# Calculate the path to the root of the project
# This assumes that `python_files` is a subdirectory in your project's root
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from enums.modelArchEnum import ModelType
#Base class for other network architectures to inherit from 
class BaseNetwork(nn.Module):
    # Initialize the network 
    def __init__(self) -> None:
        super(BaseNetwork, self).__init__()
    # Empty forward method 
    def forward(self,x: torch.Tensor) -> torch.Tensor:
         raise NotImplementedError("Forward method must be implemented by subclass")
    #Move the model to a certain device 
    def to_device(self, device: torch.device) -> None:
        self.to(device)
    # Save the model params
    def save_checkpoint(self, path:str) -> None:
        torch.save(self.state_dict(), path)
    #Load the model params 
    def load_checkpoint(self, path:str) -> None:
        self.load_state_dict(torch.load(path))

    # Get the name of the model being used
    def get_model_type(self) -> ModelType: 
        raise NotImplementedError("Get Model method must be implemented by subclass")
    
    def get_hyperparams(self) -> dict: 
        raise NotImplementedError("Hyperparameter getter must be implemented by subclass")

