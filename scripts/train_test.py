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

# Pass in command line arguments for data diretory name
# e.g. python train.py 2022-02-22-22-22
if len(sys.argv) != 2:
    print('Training script needs data!!!')
    sys.exit(1)  # exit with an error code
else:
    data_datetime = sys.argv[1]

# Designate processing unit for CNN training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")


class BearCartDataset(Dataset):
    """
    Customized dataset
    """
    def __init__(self, annotations_file, img_dir, depth_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.transform = v2.ToTensor()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #Load RGB image
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv.imread(img_path, cv.IMREAD_COLOR)
        image_tensor = self.transform(image)

        #Load depth image
        depth_img_path = os.path.join(self.depth_dir, self.img_labels.iloc[idx, 1])
        depth_image = cv.imread(depth_img_path, cv.IMREAD_GRAYSCALE)
        depth_tensor = self.transform(depth_image)

        # Combine RGB and depth images into a single tensor
        combined_tensor = torch.cat((image_tensor, depth_tensor), dim=0)

        steering = self.img_labels.iloc[idx, 2].astype(np.float32)
        throttle = self.img_labels.iloc[idx, 3].astype(np.float32)
        return combined_tensor.float(), steering, throttle


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    num_used_samples = 0
    ep_loss = 0.
    for b, (im, st, th) in enumerate(dataloader):
        target = torch.stack((st, th), dim=-1)
        feature, target = im.to(DEVICE), target.to(DEVICE)
        pred = model(feature)
        batch_loss = loss_fn(pred, target)
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
            batch_loss = loss_fn(pred, target)
            ep_loss = (ep_loss * b + batch_loss.item()) / (b + 1)
    return ep_loss


# MAIN
# Create a dataset
data_dir = os.path.join(os.path.dirname(sys.path[0]), 'data', data_datetime)
annotations_file = os.path.join(data_dir, 'labels.csv')  # the name of the csv file
img_dir = os.path.join(data_dir, 'images') # the name of the folder with all the images in it
depth_dir = os.path.join(data_dir, 'depth_images')
bearcart_dataset = BearCartDataset(annotations_file, img_dir, depth_dir)
print(f"data length: {len(bearcart_dataset)}")

# Create training dataloader and test dataloader
train_size = round(len(bearcart_dataset)*0.9)
test_size = len(bearcart_dataset) - train_size
print(f"train size: {train_size}, test size: {test_size}")
train_data, test_data = random_split(bearcart_dataset, [train_size, test_size])
train_dataloader = DataLoader(train_data, batch_size=125)
test_dataloader = DataLoader(test_data, batch_size=125)

# Create model - Pass in image size
model = convnets.DonkeyNet().to(DEVICE)  # choose the architecture class from cnn_network.py
# Hyper-parameters (lr=0.001, epochs=10 | lr=0.0001, epochs=15 or 20)
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
loss_fn = nn.MSELoss()
epochs = 15 # switch back to 15 epochs
# Optimize the model
train_losses = []
test_losses = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    ep_train_loss = train(train_dataloader, model, loss_fn, optimizer)
    ep_test_loss = test(test_dataloader, model, loss_fn)
    print(f"epoch {t+1} training loss: {ep_train_loss}, testing loss: {ep_test_loss}")
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Learning rate after scheduler step: {current_lr}")
    # save values
    train_losses.append(ep_train_loss)
    test_losses.append(ep_test_loss)
    # Apply the learning rate scheduler after each epoch
    # scheduler.step()

print("Optimize Done!")

# Graph training process
pilot_title = f'{model._get_name()}-{epochs}epochs-{lr}lr'
plt.plot(range(epochs), train_losses, 'b--', label='Training')
plt.plot(range(epochs), test_losses, 'orange', label='Test')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.ylim(0.1, max(max(train_losses), max(test_losses)))  # Set y-axis minimum to 0.1
plt.grid(True)  # Enable grid
plt.legend()
plt.title(pilot_title)
plt.savefig(os.path.join(data_dir, f'{pilot_title}.png'))
# Save the model
torch.save(model.state_dict(), os.path.join(data_dir, f'{pilot_title}.pth'))