# data.py
import numpy as np
import torch
from torchvision import datasets, transforms
from enums.datasetEnum import DatasetType

def load_dataset(args):
    # Define a default transform (will be overwritten for CIFAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset_train = None
    dataset_test = None
    dataset_img_dim = None
    num_channels = None

    if args.dataset == DatasetType.MNIST.value:
        dataset_train = datasets.MNIST(root="../data", download=True, train=True, transform=transform)
        dataset_test = datasets.MNIST(root="../data", download=True, train=False, transform=transform)
        dataset_img_dim = 28
        num_channels = 1
    elif args.dataset == DatasetType.FASHIONMNIST.value:
        dataset_train = datasets.FashionMNIST(root="../data", download=True, train=True, transform=transform)
        dataset_test = datasets.FashionMNIST(root="../data", download=True, train=False, transform=transform)
        dataset_img_dim = 28
        num_channels = 1
    elif args.dataset == DatasetType.CIFAR10.value:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset_train = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10(root="../data", train=False, transform=transform)
        dataset_img_dim = 32
        num_channels = 3
    elif args.dataset == DatasetType.CIFAR100.value:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])
        dataset_train = datasets.CIFAR100(root="../data", train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR100(root="../data", train=False, transform=transform)
        dataset_img_dim = 32
        num_channels = 3
        args.num_classes = 100
    return dataset_train, dataset_test, dataset_img_dim, num_channels

def format_dataset(dataset_data, dataset_labels):
    if isinstance(dataset_data, np.ndarray):
        formatted_data = dataset_data.astype("float32")
    elif isinstance(dataset_data, torch.Tensor):
        formatted_data = dataset_data.cpu().numpy().astype("float32")
    else:
        formatted_data = np.array(dataset_data).astype("float32")
    if formatted_data.shape[1:] == (28, 28):
        formatted_data = formatted_data.reshape(len(formatted_data), 1, 28, 28)
    elif formatted_data.shape[1:] == (32, 32, 3):
        formatted_data = formatted_data.transpose((0, 3, 1, 2))
    formatted_labels = np.array(dataset_labels).astype("int64")
    return formatted_data, formatted_labels
