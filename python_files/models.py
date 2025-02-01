# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

def get_basic_model(num_channels: int, img_size: int, num_classes: int) -> nn.Module:
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(num_channels, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            # Calculate the output size after the conv layers using a dummy input:
            dummy_input = torch.zeros(1, num_channels, img_size, img_size)
            conv_out_size = self._get_conv_output_size(dummy_input)
            self.fc1 = nn.Linear(conv_out_size, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def _get_conv_output_size(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            return int(np.prod(x.size()))

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    return SimpleCNN()

def get_resnet_model(dataset: str, num_channels: int, num_classes: int) -> nn.Module:
    if dataset in ['CIFAR10', 'CIFAR100']:
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, num_classes)
    else:
        model = models.resnet18()
        model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_model(args, num_channels: int, img_size: int, num_classes: int) -> nn.Module:
    if args.basicModel:
        return get_basic_model(num_channels, img_size, num_classes)
    else:
        return get_resnet_model(args.dataset, num_channels, num_classes)
