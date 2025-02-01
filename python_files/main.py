# main.py
from __future__ import print_function
import argparse
import logging
import os
import torch
import numpy as np
from torchvision import datasets, transforms

from losses import BlurryLoss, FocalLoss, GCELoss
from models import get_model
from data import load_dataset, format_dataset
from trainer import train_and_predict
from utils import set_seed

# Assume your custom corruption module and dataset enum are available:
from functions import DatasetCorrupterDetect as corrupter
from enums.datasetEnum import DatasetType
from cleanlab.filter import find_label_issues

SAVE_FOLDER = "plots"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

def main():
    parser = argparse.ArgumentParser(description="Modularized Training Script")
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--dataset", default="MNIST", help="Dataset to use")
    parser.add_argument("--model_architecture", default="MLP", help="Model to use")
    parser.add_argument("--num_classes", default=10, type=int, help="Number of output classes")
    parser.add_argument("--corruption_rate", default=0.0, type=float, help="Corruption rate")
    parser.add_argument("--blurry_loss_gamma", default=0.0, type=float, help="blurry loss gamma parameter, 0 == crossEntropy")
    parser.add_argument("--delay", type=int, default=0, help="Delay parameter for loss switching")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of k folds")
    parser.add_argument("--cutoff_pt", type=float, default=0.0, help="pt below which loss == 0")
    parser.add_argument("--focalLoss", action="store_true", default=False, help="Use focal loss")
    parser.add_argument("--GCELoss", action="store_true", default=False, help="Use Generalized Cross Entropy loss")
    parser.add_argument("--GCE_q", type=float, default=0.7, help="q value for GCE Loss function")
    parser.add_argument("--basicModel", action="store_true", default=False, help="Use basic model")
    parser.add_argument("--supress_print", action="store_true", default=False, help="Suppress training prints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to run for")
    args = parser.parse_args()

    # Set the seed for reproducibility
    set_seed(args.seed)

    # Load dataset and extract properties
    dataset_train, dataset_test, dataset_img_dim, num_channels = load_dataset(args)
    
    # Corrupt the training labels using your corrupter
    new_labels, corruption_tracker = corrupter.corrupt_data(dataset_train, 1, float(args.corruption_rate), args.num_classes)
    dataset_train.targets = new_labels
    dataset_data, dataset_labels = format_dataset(dataset_train.data, new_labels)

    # Select loss function based on arguments
    if args.focalLoss:
        loss_function = FocalLoss(gamma=2)
    elif args.GCELoss:
        loss_function = GCELoss(q=args.GCE_q)
    else:
        loss_function = BlurryLoss(gamma=args.blurry_loss_gamma)

    # Build the model and attach it to args (so the trainer can access it)
    model = get_model(args, num_channels, dataset_img_dim, args.num_classes)
    args.model = model

    # Train and evaluate the model
    pred_probabilities, predicted_labels, avg_accuracy = train_and_predict(
        dataset_data, dataset_labels, args, loss_function, corruption_tracker, SAVE_FOLDER, k_folds=args.num_folds, num_epochs=args.epochs
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
