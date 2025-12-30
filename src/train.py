import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DeepfakeDetectorCNN
import sys
sys.path.append('..')
from utils import AudioDataset
import time

def train_model(train_loader, val_loader, model, epochs=10, lr=0.001, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        val_acc, val_loss = evaluate_model(val_loader, model, criterion, device)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '../model/deepfake_model.pth')
            print("Model saved!")

def evaluate_model(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100. * correct / total, running_loss / len(loader)

if __name__ == "__main__":
    # Example usage (placeholders for paths)
    # train_dataset = AudioDataset('path/to/train_protocol.txt', 'path/to/audio_dir')
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_dataset = AudioDataset('path/to/val_protocol.txt', 'path/to/audio_dir')
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # model = DeepfakeDetectorCNN()
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # train_model(train_loader, val_loader, model, device=device)
    pass
