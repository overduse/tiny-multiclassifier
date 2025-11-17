import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import time
import os

# Import your custom classes
from dataset import HandDigitDataset
from model import SimpleCNN

# Training parameters
EPOCHS = 20  # Number of times to iterate over the entire dataset
BATCH_SIZE = 64 # Number of samples per batch
LEARNING_RATE = 0.001 # Optimizer's learning rate

# Data paths
TRAIN_DATA_DIR = './data/train' # Directory for the training images
MODEL_SAVE_PATH = 'saved_model_weights.pth' # Path to save the trained model weights

# Dataset split
VALIDATION_SPLIT = 0.1 # Percentage of training data to use for validation (e.g., 10%)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_model():
    """Main function to train the model."""
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor
        transforms.Normalize((0.5,), (0.5,)) # Normalize pixel values to be between -1 and 1
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
    
    # --- 4. Initialize Model, Loss Function, and Optimizer ---
    model = SimpleCNN().to(device)
    
    # Loss function: CrossEntropyLoss is suitable for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: Adam is a good default choice
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 5. Training and Validation Loop ---
    start_time = time.time()
    
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        # --- Training Phase ---
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

        # --- Validation Phase ---
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

    # --- 6. Save the Model ---
    # We save the model's 'state_dict' which contains the learned weights and biases.
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model weights saved to {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    train_model()