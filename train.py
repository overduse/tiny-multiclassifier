import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import time
import os

from dataset import HandDigitDataset
from model import SimpleCNN

import random
import numpy as np


# Training parameters
EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Data paths
TRAIN_DATA_DIR = './data/train'
MODEL_SAVE_PATH = 'saved_model_weights.pth'

# Dataset split
VALIDATION_SPLIT = 0.10
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed):
    """set all random seed to ensure the train process can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model():
    """Main function to train the model."""
    set_seed(SEED)
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        # data augment
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.8435,), (0.2694,))
    ])

    # Create the full dataset
    try:
        full_dataset = HandDigitDataset(root_dir=TRAIN_DATA_DIR, transform=transform)
        print(f"Successfully loaded dataset. Total samples: {len(full_dataset)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please make sure the '{TRAIN_DATA_DIR}' directory exists and is populated.")
        return

    # Split dataset into training and validation sets
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model, Loss Function, and Optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    start_time = time.time()
    
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            # Move data to the selected device
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad(): # No need to calculate gradients for validation
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.2f}%")

    end_time = time.time()
    training_duration = end_time - start_time
    print("\n--- Finished Training ---")
    print(f"Total training time: {training_duration / 60:.2f} minutes")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model weights saved to {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    train_model()