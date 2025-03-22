# ID Document Classification System

This system is designed to classify identity documents (ID cards and passports) by country of origin. It leverages the segmented images produced by a YOLO instance segmentation model to train classification models.

## Dataset

The dataset consists of cropped images of ID documents from 10 different countries:
- alb_id (ID Card of Albania)
- aze_passport (Passport of Azerbaijan) 
- esp_id (ID Card of Spain) 
- est_id (ID Card of Estonia) 
- fin_id (ID Card of Finland) 
- grc_passport (Passport of Greece) 
- lva_passport (Passport of Latvia) 
- rus_internalpassport (Internal passport of Russia) 
- srb_passport (Passport of Serbia) 
- svk_id (ID Card of Slovakia) 

Each class contains approximately 100 images.

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


## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Albumentations
- NumPy
- Matplotlib
- scikit-learn
- OpenCV
- tqdm 