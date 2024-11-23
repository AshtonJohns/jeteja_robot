import torch
import torch.nn as nn

class DonkeyNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Updated input channels to 4 for RGB and LiDAR
        self.conv24 = nn.Conv2d(4, 24, kernel_size=(5, 5), stride=(2, 2))  # 4 for RGB/LiDAR or RGB/Depth
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        # Fully connected layers
        self.fc1 = None  # Placeholder to be set later
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # 2 outputs: steering and throttle

        # Activation and flattening
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Initialize layers based on input size
        self._initialize_fc1()

    def _initialize_fc1(self):
        # Create a dummy input to pass through the convolution layers and calculate the output size
        dummy_input = torch.zeros(1, 4, 120, 160)  # 4 channels (RGB+LiDAR), 120x160 resolution
        dummy_output = self._forward_conv(dummy_input)
        
        # Calculate the size of the output after the convolutional layers
        num_features = dummy_output.numel()  # Total number of elements in the tensor
        self.fc1 = nn.Linear(num_features, 128)  # Set input size for the fully connected layer

    def _forward_conv(self, x):
        x = self.relu(self.conv24(x))
        x = self.relu(self.conv32(x))
        x = self.relu(self.conv64_5(x))
        x = self.relu(self.conv64_3(x))
        return x

    def forward(self, x):
        # Pass through the convolutional layers
        x = self._forward_conv(x)
        
        # Flatten the output from the convolutions
        x = self.flatten(x)
        
        # Pass through the fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
