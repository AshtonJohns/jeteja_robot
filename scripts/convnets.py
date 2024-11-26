import torch.nn as nn
import torch

class DonkeyNet(nn.Module):

    def __init__(self):
        super().__init__()
        # Convolution layers for processing image data (RGB + Depth)
        self.conv24 = nn.Conv2d(4, 24, kernel_size=(5, 5), stride=(2, 2))  # RGB/Depth input (4 channels)
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        # Fully connected layers
        # The input size to fc1 is now increased by 6 (for IMU data)
        self.fc1 = nn.Linear(64 * 8 * 13 + 6, 128)  # Added 6 for IMU data (3 accelerometer + 3 gyroscope)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # Output for two values: steering and throttle

        # Activation function and flattening
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x, imu_data):  # Input of shape (batch_size, 4, 120, 160), imu_data (batch_size, 6)
        # Process the image data through the convolutional layers
        x = self.relu(self.conv24(x))  # (120x160) -> (58x78)
        x = self.relu(self.conv32(x))  # (58x78) -> (27x37)
        x = self.relu(self.conv64_5(x))  # (27x37) -> (12x17)
        x = self.relu(self.conv64_3(x))  # (12x17) -> (10x15)
        x = self.relu(self.conv64_3(x))  # (10x15) -> (8x13)

        # Flatten the output from the convolution layers
        x = self.flatten(x)

        # Concatenate the IMU data with the image data (after convolution)
        x = torch.cat((x, imu_data), dim=1)  # Concatenate along the feature dimension (dim=1)

        # Fully connected layers
        x = self.relu(self.fc1(x))  # (8x13x64 + 6) -> (128)
        x = self.relu(self.fc2(x))  # (128) -> (128)
        x = self.fc3(x)  # Final output layer (steering and throttle)
        return x
