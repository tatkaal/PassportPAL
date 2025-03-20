import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from datetime import datetime
from pathlib import Path

# Import custom modules
from dataset import load_data, visualize_dataset_samples, visualize_augmentations
from models import CustomCNNModel, create_efficient_net_b0, create_resnet50, create_vision_transformer
from train import train_model, evaluate_model, visualize_training_history, visualize_misclassified_samples
from train import count_parameters, model_summary, compare_models

def set_seed(seed=42):
    """
    Set seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_available_device():
    """
    Get the available device (CUDA or CPU).
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")
    
    return device

def train_custom_cnn_model(data_loaders, class_names, device, args):
    """
    Train a custom CNN model from scratch.
    
    Args:
        data_loaders: Dictionary containing data loaders
        class_names: List of class names
        device: Device to train on
        args: Command-line arguments
        
    Returns:
        model: Trained model
        history: Training history
        metrics: Evaluation metrics
    """
    print("\n" + "=" * 50)
    print("Training Custom CNN Model from Scratch")
    print("=" * 50)
    
    # Create model
    model = CustomCNNModel(num_classes=len(class_names))
    model.to(device)
    
    # Print model summary
    model_summary(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create checkpoint directory
    model_name = f"custom_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = Path(args.checkpoint_dir) / model_name
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train model
    model, history = train_model(
        model=model,
        dataloaders={
            'train': data_loaders['train_loader'],
            'val': data_loaders['val_loader']
        },
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        save_dir=checkpoint_dir,
        model_name=model_name,
        early_stopping_patience=args.patience
    )
    
    # Visualize training history
    visualize_training_history(history)
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    metrics = evaluate_model(model, data_loaders['test_loader'], device, class_names)
    
    # Visualize misclassified samples
    visualize_misclassified_samples(model, data_loaders['test_loader'], class_names, device)
    
    return model, history, metrics

def train_pretrained_model(data_loaders, class_names, device, args):
    """
    Train a pretrained model (EfficientNet B0).
    
    Args:
        data_loaders: Dictionary containing data loaders
        class_names: List of class names
        device: Device to train on
        args: Command-line arguments
        
    Returns:
        model: Trained model
        history: Training history
        metrics: Evaluation metrics
    """
    print("\n" + "=" * 50)
    print(f"Training {args.pretrained_model} with Transfer Learning")
    print("=" * 50)
    
    # Create model based on selection
    if args.pretrained_model == 'efficientnet':
        model = create_efficient_net_b0(
            num_classes=len(class_names),
            pretrained=True,
            freeze_backbone=args.freeze_backbone
        )
    elif args.pretrained_model == 'resnet50':
        model = create_resnet50(
            num_classes=len(class_names),
            pretrained=True,
            freeze_backbone=args.freeze_backbone
        )
    elif args.pretrained_model == 'vit':
        model = create_vision_transformer(
            num_classes=len(class_names),
            pretrained=True,
            freeze_backbone=args.freeze_backbone
        )
    else:
        raise ValueError(f"Invalid pretrained model: {args.pretrained_model}")
    
    model.to(device)
    
    # Print model summary
    model_summary(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create checkpoint directory
    model_name = f"{args.pretrained_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = Path(args.checkpoint_dir) / model_name
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train model
    model, history = train_model(
        model=model,
        dataloaders={
            'train': data_loaders['train_loader'],
            'val': data_loaders['val_loader']
        },
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        save_dir=checkpoint_dir,
        model_name=model_name,
        early_stopping_patience=args.patience
    )
    
    # Visualize training history
    visualize_training_history(history)
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    metrics = evaluate_model(model, data_loaders['test_loader'], device, class_names)
    
    # Visualize misclassified samples
    visualize_misclassified_samples(model, data_loaders['test_loader'], class_names, device)
    
    return model, history, metrics

def main(args):
    """
    Main function to run the training pipeline.
    """
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get available device
    device = get_available_device()
    
    # Load data
    print("\nLoading data...")
    data_loaders = load_data(
        data_dir=args.data_dir,
        img_size=args.img_size,
        val_split=args.val_split,
        test_split=args.test_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    class_names = data_loaders['class_names']
    
    # Visualize dataset samples
    if args.visualize_data:
        print("\nVisualizing dataset samples...")
        visualize_dataset_samples(data_loaders['train_dataset'], class_names)
        
        # Visualize augmentations on a sample image
        print("\nVisualizing augmentations...")
        # Get a sample image from the first class
        sample_dir = os.path.join(args.data_dir, class_names[0])
        sample_img = os.path.join(sample_dir, os.listdir(sample_dir)[0])
        visualize_augmentations(sample_img, data_loaders['train_dataset'].transform)
    
    # Models to evaluate
    models_data = []
    
    # Train custom CNN model if specified
    if args.custom_cnn:
        model, history, metrics = train_custom_cnn_model(data_loaders, class_names, device, args)
        models_data.append({
            'name': 'Custom CNN',
            'model': model,
            'metrics': metrics
        })
    
    # Train pretrained model if specified
    if args.pretrained:
        model, history, metrics = train_pretrained_model(data_loaders, class_names, device, args)
        models_data.append({
            'name': args.pretrained_model.capitalize(),
            'model': model,
            'metrics': metrics
        })
    
    # Compare models if multiple models were trained
    if len(models_data) > 1:
        print("\n" + "=" * 50)
        print("Model Comparison")
        print("=" * 50)
        compare_models(models_data, metrics=['accuracy', 'precision', 'recall', 'f1'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ID Document Classification")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/cropped_images',
                        help='Path to the dataset directory')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for resizing')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fraction of data used for validation')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='Fraction of data used for testing')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    
    # Model arguments
    parser.add_argument('--custom_cnn', action='store_true',
                        help='Train a custom CNN model')
    parser.add_argument('--pretrained', action='store_true',
                        help='Train a pretrained model')
    parser.add_argument('--pretrained_model', type=str, default='efficientnet',
                        choices=['efficientnet', 'resnet50', 'vit'],
                        help='Pretrained model to use')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone layers of pretrained model')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--visualize_data', action='store_true',
                        help='Visualize dataset samples and augmentations')
    
    args = parser.parse_args()
    
    # Set defaults for demonstration
    if not (args.custom_cnn or args.pretrained):
        print("No model selected. Training both custom CNN and pretrained model by default.")
        args.custom_cnn = True
        args.pretrained = True
    
    main(args) 