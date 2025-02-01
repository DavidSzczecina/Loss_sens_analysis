import torch
import timm 
from .base_network import BaseNetwork
from pathlib import Path
import sys
# Calculate the path to the root of the project
# This assumes that `python_files` is a subdirectory in your project's root
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)
from enums.modelArchEnum import ModelType

class DataEfficientImageTransformer(BaseNetwork): 
    def __init__(self, output_size = 10, image_dimensions:int = 28, colour_channels:int = 1, dropout_rate = 0.25) -> None: 
        super(DataEfficientImageTransformer, self).__init__()

        self.deit = timm.create_model(model_name = 'deit_tiny_patch16_224', pretrained = False, num_classes = output_size, img_size = image_dimensions, in_chans = colour_channels, drop_rate = dropout_rate)

        self.hyperparams = {
            "input_channels": colour_channels, 
            "image_dimensions": image_dimensions, 
            "dropout_rate": dropout_rate, 
            "output_size": output_size
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deit(x)
        return x

    def get_hyperparams(self) -> dict:
        return self.hyperparams
    
    def get_model_type(self) -> ModelType:
        return ModelType.DEIT