from enum import Enum


class ModelType(Enum):
    MLP = "MLP"
    CNN = "CNN"
    RESNET18 = "ResNet18"
    VIT = "VIT"
    DEIT = "DEIT"
    KNN = "KNN"