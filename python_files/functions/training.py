import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
from typing import Any


# Training function
def train(
    args: Any,
    model: nn.Module,
    device: torch.device,
    train_loader_corrupted: torch.utils.data.DataLoader,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.modules.loss._Loss,
    epoch: int,
) -> float:
    # Set model to training mode
    train_loss: float = 0
    model.train()

    # Iterate over the training data from the enumerate function. Sets the variables from the returned tuple
    for batch_idx, (data, target) in enumerate(train_loader_corrupted):
        # Send the data and target to the device
        data, target = data.to(device), target.to(device)

        # Forward pass, forward method called implicitly
        output: torch.Tensor = model(data)
        # Calculate the loss using the criterion, output is the log_softmax from the forward pass, target is the GT values
        loss: torch.Tensor = criterion(output, target)
        # Zero the gradients
        optimizer.zero_grad()
        # Backpropagation
        loss.backward()
        # Update the weights using the optimizer
        optimizer.step()
        # Cumulate training loss over each batch
        train_loss += loss.item() * data.size(0)
        # Print the loss and the progress of the training

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader_corrupted.dataset),
                    100.0 * batch_idx / len(train_loader_corrupted),
                    loss.item(),
                )
            )
            if args.dry_run:
                break

    # Average the training loss, print, and return
    train_loss /= len(train_loader_corrupted.dataset)
    print("Average Training Loss: {:.6f}".format(train_loss))

    #     model.eval()
    #     eval_loss: float = 0
    #     with torch.no_grad():
    #           for batch_idx, (data, target) in enumerate(train_loader):
    #             data, target = data.to(device), target.to(device)
    #             output: torch.Tensor = model(data)
    #             loss: torch.Tensor = criterion(output, target)
    #             eval_loss += loss.item() * data.size(0)
    # #changes

    #     eval_loss /= len(train_loader.dataset)

    return train_loss
