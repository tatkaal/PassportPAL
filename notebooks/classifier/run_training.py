import os
import sys
from pathlib import Path

# Add the parent directory to the path so that we can import modules
parent_dir = Path(__file__).resolve().parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from main import main, set_seed, get_available_device
import argparse

def parse_args():
    """
    Parse command line arguments with default values.
    """
    parser = argparse.ArgumentParser(description="ID Document Classification Training")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../../data/cropped_images',
                        help='Path to the dataset directory')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for resizing')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fraction of data used for validation')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='Fraction of data used for testing')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--checkpoint_dir', type=str, default='../../checkpoints',
                        help='Directory to save model checkpoints')
    
    # Model arguments
    parser.add_argument('--custom_cnn', action='store_true', default=True,
                        help='Train a custom CNN model')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Train a pretrained model')
    parser.add_argument('--pretrained_model', type=str, default='efficientnet',
                        choices=['efficientnet', 'resnet50', 'vit'],
                        help='Pretrained model to use')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='Freeze backbone layers of pretrained model')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--visualize_data', action='store_true', default=True,
                        help='Visualize dataset samples and augmentations')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Create the checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Run the main function
    main(args) 