# %%

import pyscreenshot
import torch
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# %%


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


# %%

cnn = CNN().to("cuda")
cnn.eval()
params = []
for name, param in cnn.named_parameters():
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

try:
    cnn.load_state_dict(torch.load(f"beta_models\\best(CNN)_{params}.pth"))
    print("Model loaded")
except Exception as e:
    print("No model found")
# %%

import time

while True:
    with torch.no_grad():
        transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )
        image = pyscreenshot.grab(childprocess=False, bbox=(1514, 66, 2324, 672))
        image_tensor = transform(image).to("cuda")
        output = cnn(image_tensor.unsqueeze(0))
        output_list = output.squeeze().tolist()

        # Convertir le tenseur d'image en numpy array et transposer les dimensions pour matplotlib
        image_array = image_tensor.cpu().numpy().transpose(1, 2, 0)

        # Créer la figure et les axes
        fig, ax = plt.subplots()

        # Effacer l'axe
        ax.cla()

        # Afficher l'image
        ax.imshow(image_array)

        # Dessiner une croix rouge aux coordonnées fournies par le tenseur
        ax.plot(output_list[0] / 2, output_list[1] / 2, "rx")

        # Afficher la figure
        plt.show()

        # Pause pour 1 seconde
        time.sleep(0.1)

# %%
