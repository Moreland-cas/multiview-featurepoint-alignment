import torch
import torch.nn as nn
import torch.nn.functional as F

class OffsetPredictor(nn.Module):
    def __init__(self, input_channels, patch_size):
        super(OffsetPredictor, self).__init__()
        # Assuming input features are of shape [B, C, patch_size, patch_size]
        self.conv1 = nn.Conv2d(in_channels=input_channels * 2, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate the size of the features after convolutions and pooling
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)  # Output 2 for the offsets

    def forward(self, x1, x2):
        # Concatenate features along the channel dimension
        x = torch.cat((x1, x2), dim=1)
        
        # Pass through conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten the features
        x = x.squeeze(-1).squeeze(-1)
        
        # Pass through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        offset = 2 * torch.sigmoid(x) - 1 # (-1, 1)
        return offset


def create_model(input_channels, patch_size):
    return OffsetPredictor(input_channels, patch_size)
  