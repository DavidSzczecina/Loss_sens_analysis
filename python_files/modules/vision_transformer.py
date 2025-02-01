import torch 
from torchvision.models import vision_transformer 
from pathlib import Path
import sys
from torch import nn
# Calculate the path to the root of the project
# This assumes that `python_files` is a subdirectory in your project's root
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)
from enums.modelArchEnum import ModelType

class CustomVisionTransformer(vision_transformer.VisionTransformer):
    def __init__(self,
                 image_size: int = 28,
                 num_channels:int = 1,
                 patch_size: int = 8,
                 num_layers: int = 8,
                 num_heads: int = 16,
                 hidden_dim: int = 64,
                 mlp_dim: int = 256,
                 num_classes: int = 10,
                 dropout: float = 0.0,
                 attention_dropout: float = 0.0):
        super(CustomVisionTransformer, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes
        )

        # Modify the convolutional projection layer to accept 1-channel input
        self.conv_proj = nn.Conv2d(
            in_channels=num_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        self.hyperparams: dict = {
            "image_size": image_size, 
            "patch_size": patch_size, 
            "num_layers": num_layers, 
            "num_channels": num_channels, 
            "num_heads": num_heads, 
            "hidden_dim": hidden_dim, 
            "mlp_dim": mlp_dim, 
            "dropout": dropout, 
            "attention_dropout": attention_dropout, 
            "num_classes": num_classes, 
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def to_device(self, device: torch.device) -> None:
        self.to(device)

    def get_model_type(self) -> ModelType:
        return ModelType.VIT
    
    def get_hyperparams(self) -> dict: 
        return self.hyperparams




