# main.py
from __future__ import print_function
import argparse
import logging
import os
import torch
import numpy as np
from torchvision import datasets, transforms
from datetime import datetime

# Import loss functions 
from losses import ce, fl, bl, pz, gce, anl_ce, anl_fl

from models import get_model
from data import load_dataset, format_dataset
from trainer import train_and_predict
from utils import set_seed

# Custom corruption module and dataset enum:
from functions import DatasetCorrupterDetect as corrupter
from enums.datasetEnum import DatasetType
from cleanlab.filter import find_label_issues

# Create a timestamped folder for saving plots
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SAVE_FOLDER = os.path.join("plots", current_datetime)
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

def main():
    parser = argparse.ArgumentParser(description="Loss function Testing Script")
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--supress_print", action="store_true", default=False, help="Suppress training prints")
    
    parser.add_argument("--dataset", default="MNIST", help="Dataset to use")
    parser.add_argument("--num_classes", default=10, type=int, help="Number of output classes")
    parser.add_argument("--corruption_rate", default=0.0, type=float, help="Corruption rate")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of k folds")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to run for")
    parser.add_argument("--SGD", action="store_true", default=False, help="Use SGD optimizer")

    parser.add_argument("--delay", type=int, default=0, help="Delay parameter for loss switching")
    
    # Use a single parameter for loss selection.
    # Allowed choices: 'ce', 'focal', 'gce', 'anl_ce', 'anl_fl', 'blurry', 'pz'
    parser.add_argument("--loss", type=str, default="ce",
                        choices=["ce", "focal", "gce", "anl_ce", "anl_fl", "blurry", "pz"],
                        help="Loss function to use: 'ce' (Cross Entropy), 'focal' (Focal Loss), 'gce' (Generalized CE), "
                             "'anl_ce' (ANL Cross Entropy), 'anl_fl' (ANL Focal Loss), 'blurry' (Blurry Loss), 'pz' (Piecewise Zero Loss)")
    # Parameters specific to some losses:
    parser.add_argument("--bl_gamma", type=float, default=0.0,
                        help="Blurry loss gamma parameter (only used if --loss blurry), 0 == cross entropy")
    parser.add_argument("--pz_cutoff", type=float, default=0.0,
                        help="Cutoff value for Piecewise Zero loss (only used if --loss pz)")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load dataset and extract properties
    dataset_train, dataset_test, dataset_img_dim, num_channels = load_dataset(args)
    
    # Corrupt the training labels using your corrupter
    new_labels, corruption_tracker = corrupter.corrupt_data(
        dataset_train, 1, float(args.corruption_rate), args.num_classes
    )
    dataset_train.targets = new_labels
    dataset_data, dataset_labels = format_dataset(dataset_train.data, new_labels)
    
    # For CIFAR100, adjust the number of classes
    if args.dataset == DatasetType.CIFAR100.value:
        args.num_classes = 100
    
    # Use a dictionary to map loss names to their corresponding constructor.
    loss_dict = {
        "ce": lambda: ce(),
        "focal": lambda: fl(),
        "gce": lambda: gce(),
        "anl_ce": lambda: anl_ce(args.num_classes),
        "anl_fl": lambda: anl_fl(args.num_classes),
        "blurry": lambda: bl(args.bl_gamma),
        "pz": lambda: pz(args.pz_cutoff)
    }
    
    loss_function = loss_dict[args.loss]()
    
    # Build the model and attach it to args so the trainer can access it
    model = get_model(args, num_channels, dataset_img_dim, args.num_classes)
    args.model = model
    
    # Train and evaluate the model
    pred_probabilities, predicted_labels, avg_accuracy = train_and_predict(
        dataset_data, dataset_labels, args, loss_function, corruption_tracker, SAVE_FOLDER,
        k_folds=args.num_folds, num_epochs=args.epochs
    )
    
    print("Total predicted probabilities:", len(pred_probabilities))
    
    # Use cleanlab to find label issues
    pruning_filter = "both"
    label_issues = find_label_issues(
        dataset_labels,
        pred_probabilities,
        return_indices_ranked_by="self_confidence",
        filter_by=pruning_filter,
        frac_noise=1,
    )
    print(f"Detected label issues: {len(label_issues)}")
    
    amount_corrupted = sum(corruption_tracker)
    corrupted_labels = np.where(corruption_tracker)
    overlap = np.intersect1d(corrupted_labels, label_issues)
    
    print("Corrupted:", amount_corrupted)
    print("Detected:", len(label_issues))
    print("Overlapping indices:", len(overlap))
    
    precision = len(overlap) / len(label_issues) if len(label_issues) > 0 else 0
    recall = len(overlap) / amount_corrupted if amount_corrupted > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"F1 Score: {f1_score}")

if __name__ == "__main__":
    main()
