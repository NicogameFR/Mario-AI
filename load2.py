# pytorch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# torchsummary for easy checking and debugging
from torchsummary import summary

# torchvision for downloading and processing dataset
from torchvision import transforms, datasets

from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image


import time
import matplotlib.pyplot as plt

# Checking is CUDA available on current machine
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Project running on device: ", DEVICE)


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(128 * 10 * 10, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = x.view(-1, 128 * 10 * 10)
        x = F.relu(self.linear(x))
        return x


encoder = Encoder().to(DEVICE)
encoder.load_state_dict(torch.load("./.data/mario_data/encoder.pth"))


#%%

img = Image.open("./.data/mario_data/screenshots/0.png")
img = img.resize((160, 160))
img = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

result = encoder(img).detach().cpu().numpy()
print(result)
# %%
for i in range(10):
    img = Image.open(f"./.data/mario_data/screenshots/{i}.png")
    img = img.resize((160, 160))
    img = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

    result = encoder(img).detach().cpu().numpy()
    print(result)
# %%
