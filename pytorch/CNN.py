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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%

# Charger les données et les transformer en tenseurs
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
    ]
)


def load_data(PATH_JSON, PATH_PNG, transform=transform):
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
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    affichage(data_loader)
    return data_loader


# %%
def affichage(data_loader):
    print("Taille du dataset :", len(data_loader.dataset))
    print("Taille du batch :", data_loader.batch_size)
    print("Nombre total de batches :", len(data_loader))

    # Afficher les caractéristiques de chaque batch
    for i, batch in enumerate(data_loader, 1):
        if i == 2:
            print("Caractéristiques du batch", i, ":")
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
# Définir les hyperparamètres

# Charger les données d'entraînement et de test
train_loader = load_data(
    PATH_PNG="..\.data\Mario_data\Train\screenshots",
    PATH_JSON="..\.data\Mario_data\Train\labels\Mario_World.json",
    transform=transform,
)


# %%
# Définir l'architecture du réseau CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Constraints for layer 1
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
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
        self.fc1 = nn.Linear(3200, 1280)
        self.dropout = nn.Dropout(p=0.25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1280, 2)

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
        # Linear Layer
        out = self.fc1(out)
        # out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


batch_size = 32
num_epochs = 1000
learning_rate = 0.001

# Initialiser le modèle
model = CNN().to("cuda")
params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        params.append(name)
        params.append(str(param.data.shape))
params = "_".join(params)
params = params.replace(" ", "")
params = params.replace("_torch.Size(", "")
params = params.replace(")", "")
params = params.replace("_", "")
params = params.replace(".weight", "")
params = params.replace(".bias", "")
print(params)
try:
    torchsummary.summary(model, (3, 128, 128))
except Exception as e:
    print(e)
try:
    model.load_state_dict(torch.load(f"..\\beta_models\\best(CNN)_{params}.pth"))
    print("Model loaded")
except Exception as e:
    print(e)
    print("No model found / Switching to new model")

# %%


# Définir la fonction de perte et l'optimiseur
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_loss = 10e10
# Entraînement du modèle
for epoch in range(num_epochs):
    max_loss = []
    running_loss = 0.0
    diff_x = 0.0
    diff_y = 0.0
    diff_z = 0.0
    model.train()
    for y, [images, labels] in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        max_loss.append(loss.item())
        for i in range(batch_size):
            x = outputs[i].squeeze().tolist()[0] - labels[i].squeeze().tolist()[0]
            y = outputs[i].squeeze().tolist()[1] - labels[i].squeeze().tolist()[1]
            diff_x += abs(x)
            diff_y += abs(y)
    echec = [
        round(diff_x / (batch_size * len(train_loader)), 4),
        round(diff_y / (batch_size * len(train_loader)), 4),
    ]
    average_loss = running_loss / len(train_loader)
    max_loss = max(max_loss)
    print(f"Epoch [{epoch+1:0004d}/{num_epochs}]\r", end="")
    if average_loss < best_loss:
        best_loss = average_loss
        print(f"\nLoss: {average_loss:.4f}, Max Loss: {max_loss:.4f}, Diff: {echec}")
        torch.save(
           model.state_dict(),
           f"..\\beta_models\\best(CNN)_{params}.pth",
        )

# %%
