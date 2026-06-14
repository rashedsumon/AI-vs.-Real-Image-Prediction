import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAKEClassifier(nn.Module):
    """
    Lightweight Deep CNN optimized for processing CIFAKE's 32x32 images.
    Classifies whether an image is 'REAL' or 'AI-Generated (FAKE)'.
    """
    def __init__(self):
        super(CIFAKEClassifier, self).__init__()
        
        # Convolutional Layers: Expecting input shape [3, 32, 32]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Max Pooling Layer halves dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        # After 3 rounds of MaxPool on 32x32 image: 32 -> 16 -> 8 -> 4. Feature map size is 4x4.
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 2) # 2 outputs: [REAL, FAKE]

    def forward(self, x):
        # Apply Convolutions, ReLU activation, and Pooling
        x = self.pool(F.relu(self.conv1(x))) # Size: [32, 16, 16]
        x = self.pool(F.relu(self.conv2(x))) # Size: [64, 8, 8]
        x = self.pool(F.relu(self.conv3(x))) # Size: [128, 4, 4]
        
        # Flatten the multi-dimensional tensor array for the dense layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected operations with dropout prevention 
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x