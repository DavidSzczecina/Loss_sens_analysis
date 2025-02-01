from torch.utils.data import Dataset, DataLoader
import random
import torch
from typing import Any, Tuple


class DatasetCorrupter(Dataset):
    # Class constructor, takes in the original dataset and the corruption rate
    def __init__(
        self, original_dataset, corruption_rate: float, num_classes: int, seed: int
    ) -> None:
        self.original_dataset: Any = original_dataset
        self.num_classes: int = num_classes
        self.corruption_rate: float = corruption_rate
        self.seed: int = seed
        # Set the seed here, in the initialization, not on each __getitem__ call
        random.seed(self.seed)

    # returns the length of the dataset
    def __len__(self) -> int:
        return len(self.original_dataset)

    # returns the image and the label, the label is modified based on the corruption rate
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, old_label = self.original_dataset[idx]
        # Initialize new_label to the old_label by default
        new_label: int = old_label
        # corruption_rate chance for the label to be modified to something else
        if random.random() < self.corruption_rate:
            new_label = random.randint(0, self.num_classes - 1)
            while new_label == old_label:
                new_label = random.randint(0, self.num_classes - 1)

        return image, torch.tensor(new_label, dtype=torch.long)
