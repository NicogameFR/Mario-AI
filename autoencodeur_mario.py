# %%
from torchvision import transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader

# torchsummary for easy checking and debugging
from torchsummary import summary

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Project running on device: ", DEVICE)

# configurations for the task
config = {
    "batch_size": 32,
    "epochs": 1000,
    "lr": 1e-4,  # learning rate
}

PATH_JSON = "./mario_data/labels/Mario_World.json"
PATH_PNG = "./mario_data/screenshots/"

# %%
with open(PATH_JSON, "r") as f:
    labels = json.load(f)


transform = transforms.Compose(
    [
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ]
)

images = []
target = []

for i, filename in enumerate(os.listdir(PATH_PNG)):

    if filename.endswith(".png"):
        img = Image.open(os.path.join(PATH_PNG, filename))
        img = transform(img)
        images.append(img)
        label = labels[i]
        target.append(label)


images_tensor = torch.stack(images)
target_tensor = torch.tensor(target)

dataset = TensorDataset(images_tensor, target_tensor)


data_loader = DataLoader(
    dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
)


# %%
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(128 * 8 * 8, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.linear(x))
        return x


encoder = Encoder().to(DEVICE)

# %%
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(encoder.parameters(), lr=config["lr"], weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()


# %%
def train(model, dataloader, criterion, optimizer, save_distrib=False):
    model.train()
    train_loss = 0.0
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        x = batch[0].to(DEVICE)
        with torch.cuda.amp.autocast():
            x = batch[0].to(DEVICE)
            y_recons = model(x)
            x = batch[1].to(DEVICE)
            loss = criterion(y_recons, x)
        train_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache()
        del x, y_recons
    train_loss /= len(dataloader)
    return train_loss


# %%

for epoch in range(config["epochs"]):
    train_loss = train(encoder, data_loader, criterion, optimizer)
    print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {train_loss:.4f}\r", end="")
# %%
