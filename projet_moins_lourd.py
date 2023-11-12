##

import os
import cv2 as cv
#
# tomes = os.listdir("./train/")
# for tome in tomes:
#    imgs = os .listdir(f"./train/{tome}/pages/")
#    os.makedirs(f"./train/{tome}/pages_BW")
#    for img in imgs:
#        try:
#            image = cv.imread(f'./train/{tome}/pages/{img}')
#            SE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
#            background = cv.morphologyEx(image, cv.MORPH_DILATE, SE)
#            image = cv.divide(image, background, scale=255)
#            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#            cv.imwrite(f"./train/{tome}/pages_BW/{img}", image)
#        except: pass

# tomes = os.listdir("./train/")
#
# with open("./images.csv", "w", newline='') as csvfile:
#    for tome in tomes:
#        imgs = os.listdir(f"./train/{tome}/pages_BW/")
#        for img in imgs:
#            csvfile.writelines(f"./train/{tome}/pages_BW/{img};./train/{tome}/pages/{img}\n")
#

##

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image, ImageReadMode
from torchinfo import summary
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as f

from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

##

BATCH_SIZE = 8
IMG_SHAPE = (256, 256)

# On crée un dataset custom à partir des images du dossier 'dataset'


class TrainDataset(Dataset):
    def __init__(self, img_csv, img_size):
        self.img_csv = pd.read_csv(img_csv, delimiter=";")
        self.transform = transforms.Resize(img_size)

    def __len__(self):
        return len(self.img_csv)

    def __getitem__(self, index):
        img_paths = self.img_csv.iloc[index, :]
        image = [read_image(img_paths[0]), read_image(img_paths[1], ImageReadMode.RGB)]
        # On reshape les données :

        image = [self.transform(k) for k in image]

        # On normalise les données
        image = [(k - 127.5) / 127.5 for k in image]
        return image[0], image[1]


training_data = TrainDataset('./images.csv', IMG_SHAPE)

# On charge notre dataset dans un dataloader
x_train = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

##


for i, (img, gray_img) in enumerate(x_train):
    plt.subplot(3, 3, 3 * i + 1)
    plt.imshow(img[0].permute(1, 2, 0), cmap='gray')
    plt.axis('off')
    plt.subplot(3, 3, 3 * i + 2)
    plt.imshow(gray_img[0].permute(1, 2, 0))
    plt.axis('off')
    plt.subplot(3, 3, 3 * i + 3)
    plt.imshow(gray_img[0].permute(1, 2, 0))
    plt.axis('off')
    if i >= 2:
        break

##


class Identity(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, padding='same')
        self.batch1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, 4, 1, padding='same')
        self.batch2 = nn.BatchNorm2d(channel)
        self.conv3 = nn.Conv2d(channel, channel, 5, 1, padding='same')
        self.batch3 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

    def forward(self, entry):
        entry = self.conv1(entry)
        entry = self.batch1(entry)
        entry = self.relu(entry)
        entry = self.conv2(entry)
        entry = self.batch2(entry)
        entry = self.relu(entry)
        entry = self.conv3(entry)
        entry = self.batch3(entry)
        entry = self.relu(entry)
        return entry


class DownScale(nn.Module):
    identity: Identity

    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.down1 = nn.Conv2d(channel_in, channel_out, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(channel_in, channel_out, 5, stride=2, padding=2)
        self.batch_down1 = nn.BatchNorm2d(channel_out)
        self.batch_down2 = nn.BatchNorm2d(channel_out)
        self.id1 = nn.Conv2d(channel_in, channel_in, 3, stride=1, padding='same')
        self.batch_id1 = nn.BatchNorm2d(channel_in)
        self.id2 = nn.Conv2d(channel_out, channel_out, 5, stride=1, padding='same')
        self.batch_id2 = nn.BatchNorm2d(channel_out)
        self.identity = Identity(channel_out)
        self.relu = nn.ReLU()

    def forward(self, entry):
        x1 = entry
        entry = self.id1(entry)
        entry = self.batch_id1(entry)
        entry = self.relu(entry)
        entry = self.down1(entry)
        entry = self.batch_down1(entry)
        entry = self.relu(entry)
        entry = self.id2(entry)
        entry = self.batch_id2(entry)
        entry = self.relu(entry)
        x1 = self.down2(x1)
        x1 = self.batch_down2(x1)
        x1 = self.relu(x1)
        entry = self.identity(entry + x1)
        return entry


down = DownScale(32, 64)
print(summary(down, (BATCH_SIZE, 32, 256, 256)))


##


class UpScale(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(channel_in, channel_out, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(channel_in, channel_out, 5, stride=2, padding=2, output_padding=1)
        self.batch_up1 = nn.BatchNorm2d(channel_out)
        self.batch_up2 = nn.BatchNorm2d(channel_out)
        self.id1 = nn.Conv2d(channel_in, channel_in, 3, stride=1, padding='same')
        self.batch_id1 = nn.BatchNorm2d(channel_in)
        self.id2 = nn.Conv2d(channel_out, channel_out, 5, stride=1, padding='same')
        self.batch_id2 = nn.BatchNorm2d(channel_out)
        self.identity = Identity(channel_out)
        self.relu = nn.ReLU()

    def forward(self, entry):
        x1 = entry
        entry = self.id1(entry)
        entry = self.batch_id1(entry)
        entry = self.relu(entry)
        entry = self.up1(entry)
        entry = self.batch_up1(entry)
        entry = self.relu(entry)
        entry = self.id2(entry)
        entry = self.batch_id2(entry)
        entry = self.relu(entry)
        x1 = self.up2(x1)
        x1 = self.batch_up2(x1)
        x1 = self.relu(x1)
        entry = self.identity(entry + x1)
        return entry


up = UpScale(64, 32)
print(summary(up, (BATCH_SIZE, 64, 64, 64)))

##


class AutoEncoder(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.down1 = DownScale(1, n)
        self.down2 = DownScale(n, n * 2)
        self.down3 = DownScale(n * 2, n * 4)
        self.down4 = DownScale(n * 4, n * 8)
        self.down5 = DownScale(n * 8, n * 16)
        self.up1 = UpScale(n * 16, n * 8)
        self.up2 = UpScale(n * 8, n * 4)
        self.up3 = UpScale(n * 4, n * 2)
        self.up4 = UpScale(n * 2, n)
        self.up5 = UpScale(n, 1)
        self.relu = nn.ReLU()

    def forward(self, entry):
        x1 = self.down1(entry)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        entry = self.up1(x5)
        entry = self.up2(entry)
        entry = self.up3(entry)
        entry = self.up4(entry)
        entry = self.up5(entry)
        return entry


model = AutoEncoder(16)
print(summary(model, (BATCH_SIZE, 1, 64, 64)))

##

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

##

loss = None
epoch = 10
for i in range(epoch):
    for j, batch in tqdm(enumerate(x_train)):

        optimizer.zero_grad()
        x, y = batch
        y_hat = model(x)
        loss = loss_function(y_hat, y) * 100
        loss.backward()
        optimizer.step()
        if j % 10 == 0:
            print(f"Loss : {loss.item()}")
            plt.subplot(3, 3, 1)
            plt.imshow(x[0].permute(1, 2, 0).detach().numpy())
            plt.title('Input Image')
            plt.axis('off')
            plt.subplot(3, 3, 2)
            plt.imshow(y[0].permute(1, 2, 0).detach().numpy())
            plt.title('Ground Truth')
            plt.axis('off')
            plt.subplot(3, 3, 3)
            plt.imshow(y_hat[0].permute(1, 2, 0).detach().numpy())
            plt.title('Predicted Image')
            plt.axis('off')
            plt.show()
    print(f"Epoch {i + 1}/{epoch} terminée, Loss : {loss.item()}")
