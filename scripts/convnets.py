import torch.nn as nn

class DonkeyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2)) #3 for RGB, 4 for RGB/Depth
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        self.fc1 = nn.Linear(64*8*13, 128)  # (64*30*30, 128) for 300x300 images
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):               #   300x300                     #  120x160
        x = self.relu(self.conv24(x))  # (300-5)/2+1 = 148     |     (120-5)/2+1 = 58   (160-5)/2+1 = 78
        x = self.relu(self.conv32(x))  # (148-5)/2+1 = 72      |     (58 -5)/2+1 = 27   (78 -5)/2+1 = 37
        x = self.relu(self.conv64_5(x))  # (72-5)/2+1 = 34     |     (27 -5)/2+1 = 12   (37 -5)/2+1 = 17
        x = self.relu(self.conv64_3(x))  # 34-3+1 = 32         |     12 - 3 + 1  = 10   17 - 3 + 1  = 15
        x = self.relu(self.conv64_3(x))  # 32-3+1 = 30         |     10 - 3 + 1  = 8    15 - 3 + 1  = 13

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
