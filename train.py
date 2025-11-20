import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms

import time
import os
import sys
import logging

from dataset import HandDigitDataset
from model import SimpleCNN, BNCNN

import random
import numpy as np


# Training parameters
EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 0.0001

# Data paths
TRAIN_DATA_DIR = './data/train'
MODEL_SAVE_PATH = 'saved_model_weights.pth'
FINAL_MODEL_PATH = 'best_weights.pth'
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'

# Dataset split
VALIDATION_SPLIT = 0.20
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

def setup_logger(save_dir: str):
    """
    Set up logger: print output information to screen and .log file.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    time_str = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_dir, f'train_{time_str}.log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


def train_model():
    """Main function to train the model."""
    set_seed(SEED)

    logger, log_file_path = setup_logger(LOG_DIR)
    logger.info(f"Using device: {device}")
    logger.info(f"Log file saved to: {log_file_path}")

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        logger.info(f"Created checkpoint directory: {CHECKPOINT_DIR}")

    # Define image transformations
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.8435,), (0.2694,))
    ])

    # valset transformation
    val_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.8435,), (0.2694,))
    ])


    try:
        dataset_for_train = HandDigitDataset(root_dir=TRAIN_DATA_DIR, transform=train_transform)
        dataset_for_val = HandDigitDataset(root_dir=TRAIN_DATA_DIR, transform=val_transform)
        logger.info(f"Successfully loaded dataset. Total samples: {len(dataset_for_train)}")
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return

    dataset_size = len(dataset_for_train)
    indices = list(range(dataset_size))
    split = int(np.floor(VALIDATION_SPLIT * dataset_size))
    
    np.random.shuffle(indices)
    
    # split index
    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = Subset(dataset_for_train, train_indices)
    val_dataset = Subset(dataset_for_val, val_indices)

    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize Model, Loss Function, and Optimizer
    # model = SimpleCNN().to(device)
    model = BNCNN().to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0
    best_epoch = 0
    
    start_time = time.time()
    
    logger.info("\n--- Starting Training ---")
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
            loss = loss_fn(outputs, labels)
            
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
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.2f}%")

        epoch_save_path = os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), epoch_save_path)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f" -> New Best Model! Saved to {best_model_path}")

    end_time = time.time()
    training_duration = end_time - start_time
    logger.info("\n--- Finished Training ---")
    logger.info(f"Total training time: {training_duration / 60:.2f} minutes")
    logger.info(f"Best Validation Accuracy: {best_val_accuracy:.2f}% at Epoch {best_epoch}")

    logger.info(f"Loading best model from epoch {best_epoch}...")
    best_model_state = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
    torch.save(best_model_state, FINAL_MODEL_PATH)
    logger.info(f"Final best model weights saved to root: {FINAL_MODEL_PATH}")


if __name__ == '__main__':
    train_model()