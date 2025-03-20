import os
import argparse
import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models import CustomCNNModel, create_efficient_net_b0, create_resnet50, create_vision_transformer
from pathlib import Path
import glob
from tqdm import tqdm
import json
import time

def load_model(model_path, model_type, num_classes, device):
    """
    Load a trained model from a checkpoint file.
    
    Args:
        model_path (str): Path to the model checkpoint
        model_type (str): Type of model (custom_cnn, efficientnet, resnet50, vit)
        num_classes (int): Number of output classes
        device (torch.device): Device to load the model on
        
    Returns:
        model (nn.Module): Loaded model
    """
    # Create model based on type
    if model_type == 'custom_cnn':
        model = CustomCNNModel(num_classes=num_classes)
    elif model_type == 'efficientnet':
        model = create_efficient_net_b0(num_classes=num_classes)
    elif model_type == 'resnet50':
        model = create_resnet50(num_classes=num_classes)
    elif model_type == 'vit':
        model = create_vision_transformer(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def prepare_image(image_path, transform):
    """
    Load and preprocess an image for inference.
    
    Args:
        image_path (str): Path to the image
        transform: Image transformation pipeline
        
    Returns:
        image_tensor (torch.Tensor): Preprocessed image tensor
    """
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transformations
    transformed = transform(image=image)
    image_tensor = transformed['image']
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def get_transform(img_size=224):
    """
    Get transformation pipeline for inference.
    
    Args:
        img_size (int): Image size
        
    Returns:
        transform: Transformation pipeline
    """
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return transform

def predict_single_image(model, image_path, transform, class_names, device, show_image=False):
    """
    Predict class of a single image.
    
    Args:
        model (nn.Module): Trained model
        image_path (str): Path to the image
        transform: Image transformation pipeline
        class_names (list): List of class names
        device (torch.device): Device to run inference on
        show_image (bool): Whether to display the image with prediction
        
    Returns:
        class_name (str): Predicted class name
        confidence (float): Confidence score
    """
    # Prepare image
    image_tensor = prepare_image(image_path, transform)
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    # Get predicted class name and confidence
    predicted_idx = predicted_class.item()
    predicted_class_name = class_names[predicted_idx]
    confidence_score = confidence.item()
    
    # Display image with prediction if requested
    if show_image:
        # Load image for display
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(f"Prediction: {predicted_class_name}\nConfidence: {confidence_score:.4f}")
        plt.axis('off')
        plt.show()
    
    return predicted_class_name, confidence_score

def predict_batch(model, image_dir, transform, class_names, device, top_k=1):
    """
    Predict classes for all images in a directory.
    
    Args:
        model (nn.Module): Trained model
        image_dir (str): Directory containing images
        transform: Image transformation pipeline
        class_names (list): List of class names
        device (torch.device): Device to run inference on
        top_k (int): Number of top predictions to return
        
    Returns:
        results (dict): Dictionary containing predictions for each image
    """
    # Get image paths
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
    
    # Check if any images were found
    if not image_paths:
        print(f"No images found in {image_dir}")
        return {}
    
    # Initialize results dictionary
    results = {}
    
    # Process each image
    for image_path in tqdm(image_paths, desc="Processing images"):
        # Prepare image
        image_tensor = prepare_image(image_path, transform)
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-k predictions
        confidences, predictions = torch.topk(probabilities, top_k)
        
        # Convert to Python native types
        confidences = confidences.squeeze().cpu().numpy().tolist()
        predictions = predictions.squeeze().cpu().numpy().tolist()
        
        # Handle case where top_k=1
        if top_k == 1:
            confidences = [confidences]
            predictions = [predictions]
        
        # Format predictions
        formatted_predictions = []
        for idx, conf in zip(predictions, confidences):
            formatted_predictions.append({
                'class': class_names[idx],
                'confidence': conf
            })
        
        # Store results
        relative_path = os.path.relpath(image_path, image_dir)
        results[relative_path] = formatted_predictions
    
    return results

def main(args):
    """
    Main function for inference.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load class names
    if args.class_names_file:
        with open(args.class_names_file, 'r') as f:
            class_names = json.load(f)
    else:
        # Use default class names from the dataset directory
        class_names = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Get transformation
    transform = get_transform(args.img_size)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args.model_type, num_classes, device)
    print("Model loaded successfully.")
    
    # Perform inference
    if args.image_path:
        # Single image inference
        print(f"Performing inference on {args.image_path}...")
        predicted_class, confidence = predict_single_image(
            model, args.image_path, transform, class_names, device, show_image=True
        )
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
    
    elif args.batch_dir:
        # Batch inference
        print(f"Performing batch inference on {args.batch_dir}...")
        start_time = time.time()
        results = predict_batch(
            model, args.batch_dir, transform, class_names, device, top_k=args.top_k
        )
        end_time = time.time()
        
        # Calculate stats
        num_images = len(results)
        total_time = end_time - start_time
        avg_time = total_time / num_images if num_images > 0 else 0
        
        print(f"Processed {num_images} images in {total_time:.2f} seconds ({avg_time:.4f} seconds per image)")
        
        # Save results to file if specified
        if args.output_file:
            output_path = Path(args.output_file)
            os.makedirs(output_path.parent, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            print(f"Results saved to {output_path}")
            
        # Print sample results
        if results:
            print("\nSample predictions:")
            for path, preds in list(results.items())[:5]:
                top_pred = preds[0]
                print(f"{path}: {top_pred['class']} ({top_pred['confidence']:.4f})")
            print("...")
    
    else:
        print("Please provide either an image path for single inference or a directory for batch inference.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ID Document Classification Inference")
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['custom_cnn', 'efficientnet', 'resnet50', 'vit'],
                        help='Type of model to load')
    parser.add_argument('--data_dir', type=str, default='data/cropped_images',
                        help='Path to the dataset directory (for class names)')
    parser.add_argument('--class_names_file', type=str, default=None,
                        help='Path to a JSON file containing class names')
    
    # Input arguments
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to a single image for inference')
    parser.add_argument('--batch_dir', type=str, default=None,
                        help='Directory containing images for batch inference')
    
    # Output arguments
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save batch inference results as JSON')
    
    # Other arguments
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for inference')
    parser.add_argument('--top_k', type=int, default=1,
                        help='Number of top predictions to return for each image')
    
    args = parser.parse_args()
    
    main(args) 