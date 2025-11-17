import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """Simple CNN model"""

    def __init__(self):
        """
        define each layer of the model.
        """
        super(SimpleCNN, self).__init__()
        
        
        # input: 1x32x32
        # kernel: 5x5, output_channels: 6
        # output: 6x28x28 (32-5+1 = 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        
        # input: 6x28x28
        # pooling kernel: 2x2, 步长: 2
        # output: 6x14x14
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2nd Conv layer
        # input: 6x14x14
        # kernel: 5x5, output_channels: 16
        # output: 16x10x10 (14-5+1 = 10)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # 2nd pooling reuse self.pool
        # input: 16x10x10
        # output: 16x5x5
        
        # full connection layer
        # feed into full connection layer after faltten
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        """
        define the path of tensor when forward propagating.
        
        Args:
            x (torch.Tensor): input image Tensor, shape with  (batch_size, 1, 32, 32)
        
        Returns:
            torch.Tensor: logits shape with (batch_size, 10)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

if __name__ == '__main__':
    model = SimpleCNN()
    
    print("model structure:")
    print(model)
    
    dummy_input = torch.randn(4, 1, 32, 32)
    
    output = model(dummy_input)
    
    print("\n--- test forward propagation ---")
    print(f"input Tensor shape: {dummy_input.shape}")
    print(f"output Tensor shape: {output.shape}")
    
    assert output.shape == (4, 10), "shape of output Tensor is not correct!"
    print("\nTest of model structure and forward propagation pass！")