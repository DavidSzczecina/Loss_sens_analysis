from enum import Enum

class DatasetType(Enum):
    MNIST = "MNIST"
    CleanMNIST = "CleanMNIST"
    FASHIONMNIST = "FashionMNIST"
    KMNIST = "KMNIST"
    CIFAR10 = "CIFAR10"
    IMAGENET = "IMAGENET"
    CIFAR100 = "CIFAR100"