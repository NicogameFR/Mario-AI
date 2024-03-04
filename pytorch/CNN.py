# %%

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
import json
from PIL import Image
import os
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import torchsummary
import time

# %%

# Charger les données et les transformer en tenseurs


def load_data(PATH_JSON, PATH_PNG, transform):
    with open(PATH_JSON, "r") as f:
        labels = json.load(f)
    labels = [[label[1], label[2]] for label in labels]
    target = []
    images = []

    for i, filename in enumerate(os.listdir(PATH_PNG)):
        print(f"Loading image {i+1}/{len(os.listdir(PATH_PNG))}\r", end="")
        img = Image.open(os.path.join(PATH_PNG, filename))
        img = transform(img)
        images.append(img)
        target.append([float(labels[i][0]), float(labels[i][1])])

    images_tensor = torch.stack(images)
    target_tensor = torch.tensor(target)
    dataset = TensorDataset(images_tensor, target_tensor)
    data_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    # affichage(data_loader)
    os.system("cls")
    return data_loader


# %%
def affichage(data_loader):
    print("Taille du dataset :", len(data_loader.dataset))
    print("Taille du batch :", data_loader.batch_size)
    print("Nombre total de batches :", len(data_loader))
    batch = data_loader.dataset[0]
    print("Caractéristiques du batch", 1, ":")
    images, labels = batch
    print("Taille des images :", images.size())
    print("Taille des étiquettes :", labels.size())
    print("Affichage des images du batch")
    grid = make_grid(images, nrow=8)
    plt.figure(figsize=(15, 15))
    # add labels
    for i, (img, lbl) in enumerate(zip(images, labels)):
        plt.text(
            (i % 8) * 128,
            (i // 8) * 128,
            str(lbl.numpy()),
            fontsize=12,
            color="black",
        )
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()
    print("\n")


# %%
# Entraînement du modèle
def train_model(model, train_loader, criterion, optimizer) -> float:
    running_loss = 0.0
    model.train()
    for i, [images, labels] in enumerate(train_loader):
        images, labels = images.to("cuda"), labels.to("cuda")
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    running_loss = running_loss / len(train_loader)
    return running_loss


# %%
# Validation du modèle
def validation(model, vali_loader, criterion) -> list[float, float]:
    running_loss = 0.0
    diff_x = 0.0
    diff_y = 0.0
    diff_z = 0.0
    model.eval()
    with torch.no_grad():
        for i, [images, labels] in enumerate(vali_loader):
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            for i in range(BATCH_SIZE):
                x = outputs[i].squeeze().tolist()[0] - labels[i].squeeze().tolist()[0]
                y = outputs[i].squeeze().tolist()[1] - labels[i].squeeze().tolist()[1]
                # z = outputs[i].squeeze().tolist()[2] - labels[i].squeeze().tolist()[2]
                diff_x += abs(x)
                diff_y += abs(y)
                # diff_z += abs(z)
        echec = [
            round(diff_x / (32 * len(vali_loader)), 4),
            round(diff_y / (32 * len(vali_loader)), 4),
            # round(diff_z / (32 * len(vali_loader)), 4),
        ]
        running_loss = running_loss / len(vali_loader)
        return running_loss, echec



# %%

BATCH_SIZE = 32
IMAGE_SIZE = 128
CHANNELS = 3
epochs = 1000
learning_rate = 0.0001

# Charger les données d'entraînement et de test

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]
)


train_loader = load_data(
    PATH_PNG="..\.data\Mario_data\Train\screenshots",
    PATH_JSON="..\.data\Mario_data\Train\labels\Mario_World.json",
    transform=transform,
)
vali_loader = load_data(
    PATH_PNG="..\.data\Mario_data\Valid\screenshots",
    PATH_JSON="..\.data\Mario_data\Valid\labels\Mario_World.json",
    transform=transform,
)

# %%
# Définir l'architecture du réseau CNN


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Constraints for layer 1
        self.conv1 = nn.Conv2d(
            in_channels=CHANNELS, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.batch1 = nn.BatchNorm2d(16)

        # Constraints for layer 2
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.batch2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0
        )
        self.batch3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0
        )
        self.batch4 = nn.BatchNorm2d(128)

        # Defining the Linear layer
        self.fc1 = nn.Linear(3200, 256)
        self.dropout = nn.Dropout(p=0.15)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # defining the network flow
    def forward(self, out):
        
        # Conv 1
        
        out = self.conv1(out)
        out = self.batch1(out)
        out = self.relu(out)
        out = self.pool(out)
        # Conv 2
        
        out = self.conv2(out)
        out = self.batch2(out)
        out = self.relu(out)
        out = self.pool(out)
        # Conv 3
        
        out = self.conv3(out)
        out = self.batch3(out)
        out = self.relu(out)
        out = self.pool(out)
        # Conv 4
        
        out = self.conv4(out)
        out = self.batch4(out)
        out = self.relu(out)
        out = self.pool(out)

        out = out.view(out.size(0), -1)
        
        #out = self.dropout(out)
        out = self.fc1(out)

        out = self.relu(out)
        out = self.fc2(out)
        return out



# %%

# Initialiser le modèle
cnn = CNN().to("cuda")
try:
    torchsummary.summary(cnn, (CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
except Exception as e:
    print(e)
try:
    cnn.load_state_dict(torch.load("best(CNN).pth"))
    print("\nModel loaded :")
    for name, param in cnn.named_parameters():
        if param.requires_grad:
            print(f"{name}\t{param.data.size()}")
except Exception as e:
    print("No model found / Switching to new model")

# %%
# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(cnn.parameters(), lr=learning_rate)

best_loss : int = None


for i in range(epochs):
    running_loss_train = train_model(cnn, train_loader, criterion, optimizer)
    running_loss_valid, echec_valid = validation(cnn, vali_loader, criterion)
    os.system("cls")
    print('Epoch :', i+1, 'Train Loss :', round(running_loss_train,5), 'Valid Loss :', round(running_loss_valid,5), 'Echec :', echec_valid)

    # if running_loss_valid < best_loss:
    #    best_loss = running_loss_valid
    #    torch.save(
    #        cnn.state_dict(),
    #        f"best(CNN).pth",
    #    )
# %%
