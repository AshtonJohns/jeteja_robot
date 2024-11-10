import torch.nn as nn

class DonkeyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv24 = nn.Conv2d(4, 24, kernel_size=(5, 5), stride=(2, 2))  # Changed input channels to 4
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        self.fc1 = nn.Linear(64 * 8 * 13, 128)  # Adjusted for 120x160 input
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv24(x))
        x = self.relu(self.conv32(x))
        x = self.relu(self.conv64_5(x))
        x = self.relu(self.conv64_3(x))
        x = self.relu(self.conv64_3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
