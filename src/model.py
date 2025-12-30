import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepfakeDetectorCNN(nn.Module):
    """
    A 2D CNN architecture for Audio Deepfake Detection.
    Input: Mel-Spectrogram (1, 128, T)
    Output: Binary Classification (Real/Fake)
    """
    def __init__(self, num_classes=2):
        super(DeepfakeDetectorCNN, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully Connected Layers
        # Assuming input size is (1, 128, 126) after preprocessing
        # After 4 poolings (2x2), size becomes (256, 128/16, 126/16) = (256, 8, 7)
        self.fc1 = nn.Linear(256 * 8 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Conv Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Conv Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Conv Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def get_model():
    return DeepfakeDetectorCNN()
