import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CustomCNNModel(nn.Module):
    """
    A custom CNN model trained from scratch for ID document classification
    """
    def __init__(self, num_classes, dropout_rate=0.5):
        super(CustomCNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # Calculate input size for FC1: 256 * (224/2^4) * (224/2^4) = 256 * 14 * 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # First convolutional block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second convolutional block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third convolutional block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Fourth convolutional block
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc_bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def create_efficient_net_b0(num_classes, pretrained=True, freeze_backbone=False):
    """
    Create an EfficientNet-B0 model for ID document classification.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights or not
        freeze_backbone (bool): Freeze backbone layers or not
        
    Returns:
        nn.Module: EfficientNet model
    """
    # Load the pretrained EfficientNet model
    weights = 'DEFAULT' if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    
    # Freeze backbone layers if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    
    return model

def create_resnet50(num_classes, pretrained=True, freeze_backbone=False):
    """
    Create a ResNet-50 model for ID document classification.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights or not
        freeze_backbone (bool): Freeze backbone layers or not
        
    Returns:
        nn.Module: ResNet model
    """
    # Load the pretrained ResNet model
    weights = 'DEFAULT' if pretrained else None
    model = models.resnet50(weights=weights)
    
    # Freeze backbone layers if specified
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:  # Only fine-tune the last conv block and fc layer
                param.requires_grad = False
    
    # Replace the fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def create_vision_transformer(num_classes, pretrained=True, freeze_backbone=False):
    """
    Create a Vision Transformer model for ID document classification.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights or not
        freeze_backbone (bool): Freeze backbone layers or not
        
    Returns:
        nn.Module: Vision Transformer model
    """
    # Load the pretrained ViT model
    weights = 'DEFAULT' if pretrained else None
    model = models.vit_b_16(weights=weights)
    
    # Freeze backbone layers if specified
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "heads" not in name:
                param.requires_grad = False
    
    # Replace the classifier head
    model.heads = nn.Linear(model.hidden_dim, num_classes)
    
    return model 