import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

class IDDocumentDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, mode='train'):
        """
        ID Document Classification Dataset.
        
        Args:
            image_paths (list): List of paths to images
            labels (list): List of corresponding labels
            transform: Image augmentation transformations
            mode (str): 'train', 'val', or 'test'
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        # Use cv2 to load to ensure consistency with model training
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Get label
        label = self.labels[idx]
        
        return image, label

def get_transforms(img_size=224, use_augmentation=True):
    """
    Returns transformation pipelines for training and validation/testing.
    
    Args:
        img_size (int): Target image size
        use_augmentation (bool): Whether to use data augmentation for training
    
    Returns:
        dict: Dictionary containing transformation pipelines
    """
    # Basic transformations for all splits
    basic_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # Advanced transformations for training
    if use_augmentation:
        train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=15, p=0.5),  # Slight rotation to simulate real-world scenarios
            A.VerticalFlip(p=0.3),      # Documents can sometimes be flipped vertically
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.2),  # Simulate different camera qualities
            A.GaussNoise(mean=0, std=(0.01, 0.05), p=0.3),  # Add noise
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),  # Enhance contrast
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        train_transform = basic_transform
    
    return {
        'train': train_transform,
        'val': basic_transform,
        'test': basic_transform
    }

def load_data(data_dir, img_size=224, val_split=0.15, test_split=0.15, batch_size=32, num_workers=4, seed=42):
    """
    Load and prepare data for training, validation and testing.
    
    Args:
        data_dir (str): Directory containing class folders
        img_size (int): Target image size for resizing
        val_split (float): Fraction of data used for validation
        test_split (float): Fraction of data used for testing
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing data loaders, class names and other info
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Get class names (folder names)
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    num_classes = len(class_names)
    
    # Create class to index mapping
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
    
    # Collect image paths and labels
    image_paths = []
    labels = []
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        class_idx = class_to_idx[class_name]
        
        # Get all image files in the class directory
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(class_idx)
    
    # Split data into train, validation, and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_split, stratify=labels, random_state=seed
    )
    
    if val_split > 0:
        # Calculate the validation split from the remaining training data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=val_split/(1-test_split), 
            stratify=train_labels, random_state=seed
        )
    else:
        val_paths, val_labels = [], []
    
    print(f"Dataset split: {len(train_paths)} training, {len(val_paths)} validation, {len(test_paths)} test")
    
    # Calculate class distribution for reporting
    train_dist = np.bincount([train_labels[i] for i in range(len(train_labels))], minlength=num_classes)
    val_dist = np.bincount([val_labels[i] for i in range(len(val_labels))], minlength=num_classes) if val_paths else np.zeros(num_classes)
    test_dist = np.bincount([test_labels[i] for i in range(len(test_paths))], minlength=num_classes)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {train_dist[i]} train, {val_dist[i]} val, {test_dist[i]} test")
    
    # Get transformations
    transforms_dict = get_transforms(img_size=img_size)
    
    # Create datasets
    train_dataset = IDDocumentDataset(train_paths, train_labels, transform=transforms_dict['train'], mode='train')
    val_dataset = IDDocumentDataset(val_paths, val_labels, transform=transforms_dict['val'], mode='val') if val_paths else None
    test_dataset = IDDocumentDataset(test_paths, test_labels, transform=transforms_dict['test'], mode='test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'num_classes': num_classes,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset, 
        'test_dataset': test_dataset
    }

def visualize_dataset_samples(dataset, class_names, num_images=5, figsize=(15, 10)):
    """
    Visualize original samples from a dataset without any transformations.
    
    Args:
        dataset: PyTorch dataset object
        class_names (list): List of class names
        num_images (int): Number of images to display
        figsize (tuple): Figure size
    """
    # Select random indices
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    for i, idx in enumerate(indices):
        # Get image path and label directly from dataset
        img_path = dataset.image_paths[idx]
        label = dataset.labels[idx]
        
        # Load original image without transformations
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display image
        axes[i].imshow(image)
        axes[i].set_title(f"Class: {class_names[label]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_augmentations(img_path, transform, num_samples=5, figsize=(20, 10)):
    """
    Visualize augmentations on a single image.
    
    Args:
        img_path (str): Path to the image
        transform: Albumentation transformation pipeline
        num_samples (int): Number of augmented samples to generate
        figsize (tuple): Figure size
    """
    # Load image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples + 1, figsize=figsize)
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Show augmented images
    for i in range(num_samples):
        # Apply transformation
        augmented = transform(image=image)
        img_aug = augmented['image']
        
        # Convert tensor to numpy array for visualization
        if isinstance(img_aug, torch.Tensor):
            img_aug = img_aug.permute(1, 2, 0).numpy()
            img_aug = img_aug * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_aug = np.clip(img_aug, 0, 1)
        
        # Display augmented image
        axes[i + 1].imshow(img_aug)
        axes[i + 1].set_title(f"Augmented {i+1}")
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show() 