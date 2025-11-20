import torch
import torch.nn as nn
import torch.nn.functional as F

class BNCNN(nn.Module):
    """
    Improved CNN model with Batch Normalization and Dropout.
    """

    def __init__(self):
        super(BNCNN, self).__init__()

        self.net = nn.Sequential(
            # Input: 1x32x32
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: 32x16x16

            # --- Block 2 ---
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: 64x8x8

            nn.Flatten(),

            # --- Full Connection Layers ---
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )
        

    def forward(self, x: torch.Tensor):
        return self.net(x)


class SimpleCNN(nn.Module):
    """Simple CNN model"""

    def __init__(self):
        """
        define each layer of the model.
        """
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            # input: 1x32x32
            # kernel: 5x5, output_channels: 6 -> 6x28x28
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # pooling: 2x2 -> 6x14x14

            # input: 6x14x14
            # kernel: 5x5, output_channels: 16 -> 16x10x10
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            # pooling: 2x2 -> 16x5x5
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(), # (Batch, 16, 5, 5) -> (Batch, 400)
            # Full Connection Layers
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10)
        )
        

    def forward(self, x: torch.Tensor):
        """
        define the path of tensor when forward propagating.
        
        Args:
            x (torch.Tensor): input image Tensor, shape with  (batch_size, 1, 32, 32)
        
        Returns:
            torch.Tensor: logits shape with (batch_size, 10)
        """
        
        return self.net(x)

if __name__ == '__main__':
    model = SimpleCNN()
    # model = BNCNN()
    
    print("model structure:")
    print(model)
    
    dummy_input = torch.randn(4, 1, 32, 32)
    
    output = model(dummy_input)
    
    print("\n--- test forward propagation ---")
    print(f"input Tensor shape: {dummy_input.shape}")
    print(f"output Tensor shape: {output.shape}")
    
    assert output.shape == (4, 10), "shape of output Tensor is not correct!"
    print("\nTest of model structure and forward propagation pass")