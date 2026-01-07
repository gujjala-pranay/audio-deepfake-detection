import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DeepfakeDetectorCNN
import sys
import os
import argparse
sys.path.append('..')
from utils import AudioDataset

def train_model(train_loader, val_loader, model, epochs=10, lr=0.001, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    if not os.path.exists('../model'):
        os.makedirs('../model')
    
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
    parser = argparse.ArgumentParser(description="Train Audio Deepfake Detector")
    parser.add_argument("--train_protocol", type=str, help="Path to training protocol file")
    parser.add_argument("--train_dir", type=str, help="Path to training audio directory")
    parser.add_argument("--val_protocol", type=str, help="Path to validation protocol file")
    parser.add_argument("--val_dir", type=str, help="Path to validation audio directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples (None for full dataset)")
    
    args = parser.parse_args()
    
    if args.train_protocol and args.train_dir:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        train_dataset = AudioDataset(args.train_protocol, args.train_dir, limit=args.limit)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        val_loader = None
        if args.val_protocol and args.val_dir:
            val_dataset = AudioDataset(args.val_protocol, args.val_dir, limit=args.limit)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        model = DeepfakeDetectorCNN()
        train_model(train_loader, val_loader, model, epochs=args.epochs, lr=args.lr, device=device)
    else:
        print("Please provide --train_protocol and --train_dir to start training.")
        parser.print_help()
