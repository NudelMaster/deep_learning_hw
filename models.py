import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import torch

class CNN(nn.Module):
    def __init__(self, num_classes=10, base_width=64):
        # Note: 'base_width' corresponds to 'c1' in the original code
        super(CNN, self).__init__()
        
        c1 = base_width
        c2 = c1 * 2
        c3 = c2 * 2
        
        # Block 1: spatial / 2
        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: spatial / 2
        self.conv3 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c2)
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(c2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3: spatial / 2
        self.conv5 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(c3)
        self.conv6 = nn.Conv2d(c3, c3, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(c3)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        # Classifier
        x = self.classifier(x)
        return x

class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, num_classes=10, freeze_backbone=True, hidden_dim=512):
        super(MobileNetFeatureExtractor, self).__init__()
        # Load pre-trained MobileNetV2
        base_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        
        # Create feature extractor
        return_nodes = {
            'flatten': 'features',
        }
        self.feature_extractor = create_feature_extractor(base_model, return_nodes=return_nodes)
        
        # Freeze or unfreeze feature extractor parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze_backbone
            
        # Define fully connected layers
        self.fc1 = nn.Linear(1280, hidden_dim)
        
        # 3-layer structure: 1280 -> hidden_dim -> hidden_dim/2 -> num_classes
        dim2 = hidden_dim // 2
        
        self.classifier = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, dim2),
            nn.ReLU(),
            # classification layer
            nn.Linear(dim2, num_classes)
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)['features']
        x = self.classifier(x)
        return x

def get_model(model_name, num_classes=10, **kwargs):
    """
    Factory function to get a model instance.
    
    Args:
        model_name (str): Name of the model ('cnn', 'mobilenet', etc.)
        num_classes (int): Number of output classes
        **kwargs: Additional arguments for the model constructor
    """
    # Remove 'model_name' from kwargs if it exists to avoid multiple value errors
    if 'model_name' in kwargs:
        kwargs.pop('model_name')
        
    model_name = model_name.lower()
    
    if model_name == 'cnn':
        # Map 'c1' to 'base_width' if passed, or use 'base_width' directly
        width = kwargs.get('base_width', kwargs.get('c1', 64))
        return CNN(num_classes=num_classes, base_width=width)
        
    elif model_name == 'mobilenet':
        freeze = kwargs.get('freeze_backbone', True)
        hidden_dim = kwargs.get('hidden_dim', 512)
        return MobileNetFeatureExtractor(num_classes=num_classes, freeze_backbone=freeze, hidden_dim=hidden_dim)
    
    # Placeholders for future models
    # elif model_name == 'logreg':
    #     return LogisticRegression(num_classes=num_classes, **kwargs)
    # elif model_name == 'nn':
    #     return SimpleNN(num_classes=num_classes, **kwargs)
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
