# Audio Deepfake Detection - Training Notebook

This notebook contains the complete pipeline for training a CNN-based model for Audio Deepfake Detection using the ASVspoof 2019 dataset.

## 1. Setup and Dependencies
```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import custom modules (assuming they are in the same directory or drive)
# !cp /content/drive/MyDrive/audio-deepfake-detection/utils.py .
# !cp -r /content/drive/MyDrive/audio-deepfake-detection/src .
from utils import AudioDataset, plot_spectrogram
from src.model import DeepfakeDetectorCNN
```

## 2. Dataset Preparation
```python
# Paths (Update these to your Drive paths)
DATA_DIR = '/content/drive/MyDrive/asvspoof2019/LA/ASVspoof2019_LA_train/flac'
PROTOCOL_FILE = '/content/drive/MyDrive/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'

train_dataset = AudioDataset(PROTOCOL_FILE, DATA_DIR)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Validation set
VAL_DATA_DIR = '/content/drive/MyDrive/asvspoof2019/LA/ASVspoof2019_LA_dev/flac'
VAL_PROTOCOL = '/content/drive/MyDrive/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'

val_dataset = AudioDataset(VAL_PROTOCOL, VAL_DATA_DIR)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

## 3. Model Training
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepfakeDetectorCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    # ... training loop ...
    # (Refer to src/train.py for full implementation)
```

## 4. Evaluation & Explainability
### Why CNN?
CNNs are exceptionally good at capturing spatial hierarchies in data. By converting audio to a 2D spectrogram, we can leverage CNNs to detect:
- **Spectral discontinuities**: Artifacts at high frequencies common in TTS.
- **Harmonic inconsistencies**: Misalignment in harmonics often found in voice conversion.
- **Phase distortions**: Subtle errors in waveform reconstruction.

### Metrics
```python
# Plot Confusion Matrix
def plot_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
```
