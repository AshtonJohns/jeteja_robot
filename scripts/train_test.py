import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import convnets
import cv2 as cv
import torch.onnx  # Import ONNX support from PyTorch

# Pass in command line arguments for data directory name
if len(sys.argv) != 2:
    print('Training script needs data!!!')
    sys.exit(1)  # exit with an error code
else:
    data_datetime = sys.argv[1]

# Designate processing unit for CNN training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

def downsample_lidar(lidar_data, new_width=60, new_height=30):
    old_height, old_width = lidar_data.shape
    factor_y = old_height // new_height
    factor_x = old_width // new_width
    downsampled_data = np.zeros((new_height, new_width))
    
    for i in range(new_height):
        for j in range(new_width):
            downsampled_data[i, j] = np.mean(lidar_data[i*factor_y:(i+1)*factor_y, j*factor_x:(j+1)*factor_x])
    
    return downsampled_data

class BearCartDataset(Dataset):
    """
    Customized dataset for RGB and LiDAR combined data
    """
    def __init__(self, annotations_file, img_dir, lidar_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.lidar_dir = lidar_dir
        self.transform = v2.ToTensor()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Load RGB image
        img_name = self.img_labels.iloc[idx, 0]  # Image name from the labels.csv
        img_path = os.path.join(self.img_dir, img_name)
        image = cv.imread(img_path, cv.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Error: Could not read RGB image at {img_path}")
        image = cv.resize(image, (160, 120))  # Ensure consistent resolution (120, 160)

        # Load LiDAR data
        lidar_name = self.img_labels.iloc[idx, 3]  # LiDAR filename from the labels.csv
        lidar_path = os.path.join(self.lidar_dir, lidar_name)
        if not os.path.exists(lidar_path):
            raise FileNotFoundError(f"Error: LiDAR file {lidar_path} not found.")
        
        lidar_data = np.load(lidar_path)  # Load LiDAR .npy file
        
        # If LiDAR data is 1D (e.g., a 360-degree scan), reshape it to fit the 120x160 grid
        if lidar_data.ndim == 1:
            # Optionally, resize or interpolate LiDAR data to match the image dimensions
            lidar_data = np.resize(lidar_data, (120, 160))  # Reshape to the required resolution

        # Downsample the LiDAR data to 1800 points (30x60 grid)
        lidar_data = downsample_lidar(lidar_data)  # Downsampling to 1800 points

        # Convert images and LiDAR data to tensor format
        image_tensor = self.transform(image)
        lidar_tensor = torch.tensor(lidar_data, dtype=torch.float32).unsqueeze(0)  # Add a channel dimension for LiDAR

        # Combine RGB and LiDAR into a single tensor (4 channels)
        combined_tensor = torch.cat((image_tensor, lidar_tensor), dim=0)

        # Steering and throttle values
        steering = self.img_labels.iloc[idx, 1].astype(np.float32)
        throttle = self.img_labels.iloc[idx, 2].astype(np.float32)
        return combined_tensor.float(), steering, throttle


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    num_used_samples = 0
    ep_loss = 0.
    for b, (im, st, th) in enumerate(dataloader):
        target = torch.stack((st, th), dim=-1)
        feature, target = im.to(DEVICE), target.to(DEVICE)
        pred = model(feature)
        batch_loss = loss_fn(pred, target)  # Loss function without mask
        optimizer.zero_grad()  # zero previous gradient
        batch_loss.backward()  # back propagation
        optimizer.step()  # update params
        num_used_samples += target.shape[0]
        print(f"batch loss: {batch_loss.item()} [{num_used_samples}/{len(dataloader.dataset)}]")
        ep_loss = (ep_loss * b + batch_loss.item()) / (b + 1)
    return ep_loss


def test(dataloader, model, loss_fn):
    model.eval()
    ep_loss = 0.
    with torch.no_grad():
        for b, (im, st, th) in enumerate(dataloader):
            target = torch.stack((st, th), dim=-1)
            feature, target = im.to(DEVICE), target.to(DEVICE)
            pred = model(feature)
            batch_loss = loss_fn(pred, target)  # Loss function without mask
            ep_loss = (ep_loss * b + batch_loss.item()) / (b + 1)
    return ep_loss


# Custom loss function (standard MSE without NaN masking)
def standard_loss(output, target):
    loss = ((output - target) ** 2).mean()  # Mean Squared Error loss
    return loss


# MAIN
# Create a dataset
data_dir = os.path.join(os.path.dirname(sys.path[0]), 'data', data_datetime)
annotations_file = os.path.join(data_dir, 'labels.csv')
img_dir = os.path.join(data_dir, 'rgb_images')  # RGB images directory
lidar_dir = os.path.join(data_dir, 'lidar_images')  # LiDAR data directory
bearcart_dataset = BearCartDataset(annotations_file, img_dir, lidar_dir)
print(f"data length: {len(bearcart_dataset)}")

# Create training and test dataloaders
train_size = round(len(bearcart_dataset) * 0.9)
test_size = len(bearcart_dataset) - train_size
print(f"train size: {train_size}, test size: {test_size}")
train_data, test_data = random_split(bearcart_dataset, [train_size, test_size])
train_dataloader = DataLoader(train_data, batch_size=125)
test_dataloader = DataLoader(test_data, batch_size=125)

# Create model
model = convnets.DonkeyNet().to(DEVICE)  # Adjust input channels to 4 (RGB + LiDAR)
# Hyper-parameters
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
loss_fn = standard_loss  # Use standard loss function (no masking)
epochs = 15
train_losses = []
test_losses = []
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    ep_train_loss = train(train_dataloader, model, loss_fn, optimizer)
    ep_test_loss = test(test_dataloader, model, loss_fn)
    print(f"epoch {t + 1} training loss: {ep_train_loss}, testing loss: {ep_test_loss}")
    train_losses.append(ep_train_loss)
    test_losses.append(ep_test_loss)

print("Optimization Done!")

# Graph training process
pilot_title = f'{model._get_name()}-{epochs}epochs-{lr}lr'
plt.plot(range(epochs), train_losses, 'b--', label='Training')
plt.plot(range(epochs), test_losses, 'orange', label='Test')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.ylim(0.0, 0.1)
plt.yticks(np.arange(0, 0.15, 0.01))  # Set y-axis ticks from 0 to 0.1 in steps of 0.01
plt.grid(True)
plt.legend()
plt.title(pilot_title)
plt.savefig(os.path.join(data_dir, f'{pilot_title}.png'))

# Save the model (weights only)
torch.save(model.state_dict(), os.path.join(data_dir, f'{pilot_title}.pth'))
print("Model weights saved")

# ONNX export
dummy_input = torch.randn(1, 4, 120, 160).to(DEVICE)  # Adjust shape for 120x160 RGB-LiDAR
onnx_model_path = os.path.join(data_dir, f'{pilot_title}.onnx')
torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11)
print(f"Model exported to ONNX format at: {onnx_model_path}")
