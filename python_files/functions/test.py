import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from functions import sens_spec
import numpy as np
from numpy import ndarray
# Test function


def test(
    model: nn.Module,
    device: torch.device,
    test_loader: torch.utils.data.DataLoader,
    jobid: int,
    num_classes: int,
):
    # Set model to evaluation mode
    model.eval()
    # Initialize the loss and correct values to 0
    test_loss: float = 0

    # Create the confusion matrix
    confusion_matrix: ndarray = np.zeros((num_classes, num_classes), dtype=int)

    correct: int = 0
    # Disable gradient calculation for this context
    with torch.no_grad():
        for data, target in test_loader:
            # Send the data and target to the device
            data, target = data.to(device), target.to(device)
            # Forward pass, forward method called implicitly
            output: torch.Tensor = model(data)

            log_probabilities: torch.Tensor = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(
                log_probabilities, target, reduction="sum"
            ).item()  # sum up batch loss
            # Creates a [batch_size, 1] shape
            pred: torch.Tensor = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            # transforms target to the same shape as pred. .eq() returns boolean for equality, which is summed into correct
            correct += pred.eq(target.view_as(pred)).sum().item()
            # Creates a 1D array for predictions
            pred_arr: torch.Tensor = output.argmax(dim=1)

            # Fill the confusion matrix
            for target, pred in zip(target, pred_arr):
                confusion_matrix[target, pred] += 1

    test_loss /= len(test_loader.dataset)
    # Print the test loss and accuracy
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    avg_sens, avg_spec = sens_spec.compute(confusion_matrix, num_classes)

    return test_loss, correct, len(test_loader.dataset), avg_sens, avg_spec
