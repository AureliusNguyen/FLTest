import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, channels=3, num_classes=10):
        """
        Initialize the LeNet model.

        Args:
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
            num_classes (int): Number of output classes.
        """
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 6, kernel_size=5)  # Convolutional layer 1
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)      # Average pooling layer
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)          # Convolutional layer 2
        self.fc1 = nn.Linear(16 * 5 * 5, 120)                 # Fully connected layer 1
        self.fc2 = nn.Linear(120, 84)                         # Fully connected layer 2
        self.fc3 = nn.Linear(84, num_classes)                 # Fully connected layer 3

    def forward(self, x):
        """
        Define the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, 32, 32).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 16 * 5 * 5)            # Flatten
        x = F.relu(self.fc1(x))               # FC1 -> ReLU
        x = F.relu(self.fc2(x))               # FC2 -> ReLU
        x = self.fc3(x)                       # FC3
        return x



