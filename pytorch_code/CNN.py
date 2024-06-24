# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import numpy as np
import json
from PIL import Image
import os
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
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
taille = 2500
output_size = 2

def load_data(PATH_JSON, PATH_PNG, transform=transform):
    with open(PATH_JSON, "r") as f:
        labels = json.load(f)
    labels = [[label[1], label[2]] for label in labels]
    target = []
    images = []
    mean, std = 0, 0
    for i, filename in enumerate(os.listdir(PATH_PNG)):
        if i == taille:
            break
        print(f"Loading image {i+1}/{taille}\r", end="")
        img = Image.open(os.path.join(PATH_PNG, filename))
        img = transform(img)
        mean, std = img.mean([1, 2]), img.std([1, 2])
        normalize = transforms.Normalize(mean, std)
        img = normalize(img)
        images.append(img)
        target.append([float(labels[i][0]), float(labels[i][1])])
        #target.append([float(labels[i][0]), float(labels[i][1]), float(labels[i][2])])

    images_tensor = torch.stack(images)
    target_tensor = torch.tensor(target)

    dataset = TensorDataset(images_tensor, target_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# %%
def affichage(data_loader):
    print("Taille du dataset :", len(data_loader.dataset))
    print("Taille du batch :", data_loader.batch_size)
    print("Nombre total de batches :", len(data_loader))
    for i, batch in enumerate(data_loader, 1):
        if i == 1:
            print("Caractéristiques du batch", i, ":")
            images, labels = batch
            print("Taille des images :", images.size())
            print("Taille des étiquettes :", labels.size())
            grid = make_grid(images, nrow=8)
            plt.figure(figsize=(15, 15))
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
batch_size = 32

train_loader, validation_loader = load_data(
    PATH_PNG="..\.data\Mario_data\Train\screenshots",
    PATH_JSON="..\.data\Mario_data\Train\labels\Mario_World.json",
    transform=transform,
)

affichage(train_loader)
affichage(validation_loader)


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
        self.dropout = nn.Dropout(p=0.15)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1280, output_size)

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
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


num_epochs = 1000
learning_rate = 5e-4

# Initialiser le modèle
model = CNN().to(DEVICE)
params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        params.append(name)
        params.append(str(param.data.shape))
params = "_".join(params)
params = params.replace(" ", "")
params = params.replace("torch.Size(", "")
params = params.replace(")", "")
params = params.replace("_", "")
params = params.replace(".weight", "")
params = params.replace(".bias", "")
try:
    print(1 / 0)
    model.load_state_dict(torch.load(f"..\\beta_models\\best(CNN)_{params}.pth"))
    print("Model loaded")
except Exception as e:
    print(e)
    print("No model found / Switching to new model")

# %%

criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
best_loss = 10e10


# %%
def train_model(model, train_loader):
    max_loss = []
    running_loss = 0.0
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

    average_loss = running_loss / len(train_loader)
    max_loss = max(max_loss)
    return average_loss, max_loss


# %%


def validation_model(model, test_loader):
    model.eval()
    max_loss = []
    running_loss = 0.0
    diff_x = 0.0
    diff_y = 0.0
    diff_z = 0.0
    for y, [images, labels] in enumerate(train_loader):
        with torch.no_grad():
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item()
        max_loss.append(loss.item())
        for i in range(len(outputs)):
            x = outputs[i].squeeze().tolist()[0] - labels[i].squeeze().tolist()[0]
            y = outputs[i].squeeze().tolist()[1] - labels[i].squeeze().tolist()[1]
            #z = outputs[i].squeeze().tolist()[2] - labels[i].squeeze().tolist()[2]
            diff_x += abs(x)
            diff_y += abs(y)
            #diff_y += abs(z)
    echec = [
        round(diff_x / (len(outputs) * len(train_loader)), 4),
        round(diff_y / (len(outputs) * len(train_loader)), 4),
        #round(diff_z / (batch_size * len(train_loader)), 4),
    ]
    average_loss = running_loss / len(train_loader)
    max_loss = max(max_loss)
    return average_loss, max_loss, echec


# %%
diff = 0
for epoch in range(num_epochs):

    average_loss_train, _ = train_model(model, train_loader)
    average_loss_valid, max_loss_valid, echec = validation_model(
        model, validation_loader
    )
    print(f"Epoch [{epoch+1:0004d}/{num_epochs}] +{diff: 4d}\r", end="")
    diff += 1
    if average_loss_valid < best_loss:
        diff = 0
        best_loss = average_loss_valid
        print(f"\nLoss: {average_loss_valid:.4f}, Max Loss: {max_loss_valid:.4f}, Diff: {echec}")

