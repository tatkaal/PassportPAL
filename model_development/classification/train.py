import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
import seaborn as sns
import pandas as pd
import wandb

def train_model(model, dataloaders, criterion, optimizer, scheduler=None, 
                num_epochs=25, device='cuda', save_dir='checkpoints', 
                model_name='model', early_stopping_patience=10):
    """
    Trains a PyTorch model and tracks various metrics.
    
    Args:
        model: PyTorch model to train
        dataloaders: Dictionary containing 'train' and 'val' dataloaders
        criterion: Loss function to optimize
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler (optional)
        num_epochs: Number of epochs to train for
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save model checkpoints
        model_name: Name of the model for saving checkpoints
        early_stopping_patience: Number of epochs to wait before early stopping
        
    Returns:
        model: Best model weights
        history: Training history
    """
    since = time.time()
    
    # Create directory for saving checkpoints
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize best model weights and best accuracy
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_epoch = 0
    
    # Initialize history dictionary to track metrics
    history = {
        'train_loss': [],
        'train_acc': [], 
        'val_loss': [],
        'val_acc': []
    }
    
    # Initialize early stopping counter
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase}'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                # Track history only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Adjust learning rate using scheduler if provided
                if scheduler is not None:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(epoch_loss)
                    else:
                        scheduler.step()
                
                # Deep copy the model if best accuracy
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    best_epoch = epoch
                    
                    # Save best model checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        'acc': epoch_acc,
                    }, os.path.join(save_dir, f'{model_name}_best.pth'))
                    
                    # Reset early stopping counter
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    
            if phase == 'train':
                wandb.log({
                    'Epoch': epoch + 1,
                    'Train Loss': epoch_loss,
                    'Train Accuracy': epoch_acc.item()
                })
            else:
                wandb.log({
                    'Epoch': epoch + 1,
                    'Validation Loss': epoch_loss,
                    'Validation Accuracy': epoch_acc.item()
                })
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'acc': epoch_acc,
            }, os.path.join(save_dir, f'{model_name}_epoch_{epoch+1}.pth'))
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
            
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f} at epoch {best_epoch+1}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def evaluate_model(model, dataloader, device='cuda', class_names=None):
    """
    Evaluates a trained PyTorch model on a test set.
    
    Args:
        model: Trained PyTorch model
        dataloader: PyTorch dataloader containing test data
        device: Device to evaluate on ('cuda' or 'cpu')
        class_names: List of class names
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists to store predictions and labels
    all_preds = []
    all_labels = []
    
    # No gradient computation needed for evaluation
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate evaluation metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # Print summary
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    # Generate detailed classification report
    if class_names is not None:
        print('\nClassification Report:')
        print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    
    if class_names is not None:
        # If there are many classes, adjust the font size
        font_size = max(8, 12 - 0.4 * len(class_names))
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, annot_kws={"size": font_size})
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix')
        plt.tight_layout()
        plt.show()
    
    # Return metrics
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'true_labels': all_labels
    }

def visualize_training_history(history):
    """
    Visualize training and validation loss and accuracy.
    
    Args:
        history: Dictionary containing training and validation metrics
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot training and validation accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_misclassified_samples(model, dataloader, class_names, device='cuda', num_samples=10, title="Misclassified Samples"):
    """
    Visualize samples that are misclassified by the model in the same format as original samples visualization.
    
    Args:
        model: Trained PyTorch model
        dataloader: PyTorch dataloader containing data
        class_names: List of class names
        device: Device to evaluate on ('cuda' or 'cpu')
        num_samples: Number of misclassified samples to visualize
        title: Title for the figure
    """
    # Set model to evaluation mode
    model.eval()
    
    # Lists to store misclassified samples
    misclassified_images = []
    misclassified_true_labels = []
    misclassified_pred_labels = []
    
    # No gradient computation needed for evaluation
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Find misclassified samples
            misclassified_mask = (preds != labels)
            
            # Add misclassified samples to lists
            for i in range(inputs.size(0)):
                if misclassified_mask[i] and len(misclassified_images) < num_samples:
                    # Get the image
                    img = inputs[i].cpu().permute(1, 2, 0).numpy()
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img = np.clip(img, 0, 1)
                    
                    misclassified_images.append(img)
                    misclassified_true_labels.append(labels[i].item())
                    misclassified_pred_labels.append(preds[i].item())
            
            # Stop if we have enough samples
            if len(misclassified_images) >= num_samples:
                break
    
    # Plot misclassified images
    num_images = len(misclassified_images)
    if num_images == 0:
        print("No misclassified samples found.")
        return
    
    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
    fig.suptitle(title, fontsize=16)
    
    for i in range(num_images):
        axes[i].imshow(misclassified_images[i])
        axes[i].set_title(f"True: {class_names[misclassified_true_labels[i]]}\nPred: {class_names[misclassified_pred_labels[i]]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model, input_size=(3, 224, 224), batch_size=1):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (channel, height, width)
        batch_size: Batch size for the input tensor
    """
    # Create a dummy input tensor
    x = torch.zeros(batch_size, *input_size)
    
    # Print model summary
    print(model)
    print(f"\nNumber of trainable parameters: {count_parameters(model):,}")

def compare_models(models_data, metrics=None):
    """
    Compare multiple models based on specified metrics.
    
    Args:
        models_data: List of dictionaries containing model information
                     Each dictionary should have keys:
                     - 'name': Model name
                     - 'metrics': Dictionary containing model metrics
        metrics: List of metrics to compare (if None, use all available metrics)
    """
    if not models_data:
        print("No models provided for comparison.")
        return
    
    # If metrics not specified, use all available metrics
    if metrics is None and 'metrics' in models_data[0]:
        metrics = [m for m in models_data[0]['metrics'].keys() if isinstance(models_data[0]['metrics'][m], (int, float))]
    
    # Create a DataFrame for comparison
    comparison_data = {
        'Model': [model['name'] for model in models_data]
    }
    
    for metric in metrics:
        comparison_data[metric] = [model['metrics'].get(metric, None) for model in models_data]
    
    df = pd.DataFrame(comparison_data)
    
    # Print comparison
    print("Model Comparison:")
    print(df.to_string(index=False))
    
    # Plot comparison
    if len(metrics) > 0:
        # Melt DataFrame for easier plotting
        df_melt = pd.melt(df, id_vars=['Model'], value_vars=metrics, var_name='Metric', value_name='Value')
        
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Model', y='Value', hue='Metric', data=df_melt)
        plt.title('Model Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)  # Assuming metrics are between 0 and 1
        plt.tight_layout()
        plt.show() 