# trainer.py
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from utils import disable_print, enable_print

def plot_pred_probability_distribution(epoch, fold, epoch_pred_probs, corruption_tracker, save_folder):
    """Plot and save a stacked bar chart of predicted probabilities for corrupted vs. uncorrupted samples."""
    corrupted_probs = [prob for prob, corrupted in zip(epoch_pred_probs, corruption_tracker) if corrupted]
    uncorrupted_probs = [prob for prob, corrupted in zip(epoch_pred_probs, corruption_tracker) if not corrupted]
    bins = np.linspace(0, 1, 51)
    uncorrupted_hist, _ = np.histogram(uncorrupted_probs, bins=bins)
    corrupted_hist, _ = np.histogram(corrupted_probs, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    width = 0.9 * (bins[1] - bins[0])
    
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, uncorrupted_hist, width=width, color='blue', alpha=0.7, label='Uncorrupted')
    plt.bar(bin_centers, corrupted_hist, width=width, bottom=uncorrupted_hist, color='red', alpha=0.7, label='Corrupted')
    plt.title(f"Fold {fold}, Epoch {epoch}: Distribution of Predicted Probabilities")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{save_folder}/Epoch_{epoch}_pred_probs_stacked.png")
    plt.close()

def train_and_predict(dataset_data, dataset_labels, args, loss_function, corruption_tracker, save_folder,
                      k_folds=5, num_epochs=10, batch_size=64):
    """
    Trains the provided model using k-fold cross-validation and returns prediction probabilities,
    predicted labels, and average accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=123)
    results = {}
    total_pred_probabilities = []
    total_y_true = []
    total_y_pred = []

    # Standardized training configuration printing
    print("Training Configuration:")
    print(f"  Loss function: {args.loss}")
    print(f"  Corruption rate: {args.corruption_rate}")
    print(f"  Delay: {args.delay}")
    if args.loss == "blurry":
        print(f"  Blurry loss parameters: gamma = {args.bl_gamma}, cutoff_pt = {args.cutoff_pt}")
    elif args.loss == "pz":
        print(f"  PZ loss parameters: cutoff = {args.pz_cutoff}")
    
    if args.supress_print:
        disable_print()

    # Iterate over k folds
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset_data), 1):
        print(f"\n--- FOLD {fold} ---")
        train_subset = Subset(TensorDataset(torch.tensor(dataset_data), torch.tensor(dataset_labels)), train_ids)
        test_subset = Subset(TensorDataset(torch.tensor(dataset_data), torch.tensor(dataset_labels)), test_ids)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        model = args.model
        model.to(device)
        
        # Select optimizer based on flag
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        if args.SGD:
            optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        
        # Use a scheduler for CIFAR datasets
        if args.dataset in ['CIFAR10', 'CIFAR100']:
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1 if args.dataset == 'CIFAR10' else 0.2)

        scaler = GradScaler()
        
        for epoch in range(num_epochs):
            # Adjust loss function parameters for cutoff parameter
            if args.loss in ["blurry", "pz"]:
                if epoch >= args.delay:
                    loss_function.gamma = float(args.bl_gamma) if args.loss == "blurry" else 0.0
                    loss_function.cutoff_pt = args.cutoff_pt if args.loss == "blurry" else 0.0
                else:
                    loss_function.gamma = 0.0
                    loss_function.cutoff_pt = 0.0

            epoch_pred_probs = []
            model.train()
            running_loss = 0.0
            epoch_start_time = time.time()
            
            # Training loop
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                ground_truth_probs = probabilities[range(len(targets)), targets]
                epoch_pred_probs.extend(ground_truth_probs.cpu().detach().numpy())
            
            if args.dataset in ['CIFAR10', 'CIFAR100']:
                scheduler.step()
            
            avg_train_loss = running_loss / len(train_loader)
            
            # Evaluation loop
            model.eval()
            total_test_loss = 0.0
            fold_pred_probabilities = []
            fold_true_labels = []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = loss_function(outputs, targets)
                    total_test_loss += loss.item()
                    probabilities = torch.softmax(outputs, dim=1)
                    fold_pred_probabilities.append(probabilities.cpu().numpy())
                    fold_true_labels.append(targets.cpu().numpy())
            avg_test_loss = total_test_loss / len(test_loader)
            fold_pred_probabilities = np.concatenate(fold_pred_probabilities, axis=0)
            fold_true_labels = np.concatenate(fold_true_labels, axis=0)
            y_pred = np.argmax(fold_pred_probabilities, axis=1)
            test_accuracy = accuracy_score(fold_true_labels, y_pred) * 100
            macro_avg_loss = np.mean([avg_train_loss, avg_test_loss])
            epoch_duration = int(time.time() - epoch_start_time)
            
            print(f'Fold {fold}, Epoch {epoch+1:2d}: Train Loss: {avg_train_loss:.3f}, '
                  f'Test Loss: {avg_test_loss:.3f}, Macro Loss: {macro_avg_loss:.3f}, '
                  f'Accuracy: {test_accuracy:.2f}%, Duration: {epoch_duration} sec.')
            
            if fold == 1:
                plot_pred_probability_distribution(epoch+1, fold, epoch_pred_probs, corruption_tracker, save_folder)
        
        total_pred_probabilities.extend(list(zip(test_ids, fold_pred_probabilities)))
        total_y_true.extend(list(zip(test_ids, fold_true_labels)))
        total_y_pred.extend(y_pred)
        fold_acc = accuracy_score(fold_true_labels, y_pred)
        results[fold] = fold_acc
        print(f'Fold {fold} Accuracy: {fold_acc * 100:.2f}%')
    
    if args.supress_print:
        enable_print()
    
    avg_accuracy = np.mean(list(results.values()))
    print('--------------------------------')
    print("K-FOLD CROSS VALIDATION RESULTS:")
    for fold, acc in results.items():
        print(f"  Fold {fold}: {acc * 100:.2f}%")
    print(f"Average Accuracy: {avg_accuracy * 100:.2f}%")
    
    # Sort predictions by test set order
    total_pred_probabilities = [x[1] for x in sorted(total_pred_probabilities, key=lambda x: x[0])]
    total_y_true = [x[1] for x in sorted(total_y_true, key=lambda x: x[0])]
    
    return np.array(total_pred_probabilities), np.array(total_y_pred), avg_accuracy
