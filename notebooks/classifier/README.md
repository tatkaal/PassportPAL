# ID Document Classification System

This system is designed to classify identity documents (ID cards and passports) by country of origin. It leverages the segmented images produced by a YOLO instance segmentation model to train classification models.

## Dataset

The dataset consists of cropped images of ID documents from 10 different countries:
- alb_id: Albanian ID cards
- aze_passport: Azerbaijan passports
- esp_id: Spanish ID cards
- est_id: Estonian ID cards
- fin_id: Finnish ID cards
- grc_passport: Greek passports
- lva_passport: Latvian passports
- rus_internalpassport: Russian internal passports
- srb_passport: Serbian passports
- svk_id: Slovak ID cards

Each class contains approximately 100 images.

## Project Structure

```
notebooks/classifierlear/
├── dataset.py         # Dataset loading and preprocessing utilities
├── models.py          # Model architectures (custom CNN and transfer learning models)
├── train.py           # Training and evaluation functionality
├── main.py            # Main training pipeline
├── inference.py       # Inference utilities for using trained models
├── run_training.py    # Script for running training with default settings
└── README.md          # This file
```

## Models

The system implements two types of models:

1. **Custom CNN Model**: A model trained from scratch with the following architecture:
   - 4 convolutional blocks (each with convolution, batch normalization, ReLU, and max pooling)
   - 3 fully connected layers with dropout to prevent overfitting

2. **Transfer Learning Models**:
   - EfficientNet-B0: A lightweight but powerful convolutional neural network
   - ResNet-50: A deep residual network with 50 layers
   - Vision Transformer (ViT): A transformer-based model for image classification

## Data Preprocessing and Augmentation

The data preprocessing pipeline includes:
- Resizing images to a fixed size (default: 224x224)
- Normalization using ImageNet mean and standard deviation
- Data augmentation techniques:
  - Random rotation (up to 15 degrees)
  - Vertical flipping (no horizontal flipping as it's unrealistic for documents)
  - Color jittering (brightness, contrast, saturation, hue)
  - Gaussian blur and noise to simulate different camera qualities
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement

## Training

To train the models, run:

```bash
python run_training.py
```

or with custom arguments:

```bash
python run_training.py --data_dir /path/to/data --epochs 50 --batch_size 32
```

Key training parameters:
- Early stopping based on validation accuracy
- Cosine annealing learning rate scheduler
- Adam optimizer with weight decay for regularization
- Checkpoint saving for the best model and at regular intervals

## Evaluation

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Visualization of misclassified samples

## Inference

To run inference on a single image:

```bash
python inference.py --model_path /path/to/model.pth --model_type efficientnet --image_path /path/to/image.jpg
```

For batch inference on a directory of images:

```bash
python inference.py --model_path /path/to/model.pth --model_type efficientnet --batch_dir /path/to/images --output_file results.json
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- torchvision
- Albumentations
- NumPy
- Matplotlib
- scikit-learn
- OpenCV
- tqdm 