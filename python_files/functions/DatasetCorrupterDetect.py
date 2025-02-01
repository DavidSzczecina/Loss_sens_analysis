import torch
import random
from typing import List, Tuple


def corrupt_data(dataset, seed: int, corruption_rate: float, num_classes: int) -> Tuple:
    num_samples = len(dataset)
    corruption_tracker: List[bool] = []

    new_labels: List[int] = []
    random.seed(seed)
    for idx in range(num_samples):
        image, label = dataset[idx]
        if random.random() < corruption_rate:
            corruption_tracker.append(1)
            new_label = random.randint(0, num_classes - 1)
            while new_label == label:
                new_label = random.randint(0, num_classes - 1)
            new_labels.append(new_label)
        else:
            corruption_tracker.append(0)
            new_labels.append(label)

    return torch.tensor(new_labels, dtype=torch.long), corruption_tracker
