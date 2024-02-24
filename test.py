# %% [markdown]
# Librairies
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
from torchsummary import summary
import numpy as np
import sys

# %% [markdown]
# Configuration


def variables(SIZE, LAYERS, LATENT_DIM):
    global CHANNELS, KERNELS, STRIDES, PADDING, LINEAR_DIM, DEVICE
    CHANNELS = [3]
    KERNELS = [3 for i in range(LAYERS)]
    CHANNELS.extend([2 ** (i + 5) for i in range(LAYERS)])
    STRIDES = [2 for i in range(LAYERS)]
    PADDING = [1 for i in range(LAYERS)]
    LINEAR_DIM = int((SIZE / (2**LAYERS)) ** 2 * CHANNELS[-1])

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# %% [markdown]
# Load data


def load_data(PATH_JSON, PATH_PNG):
    with open(PATH_JSON, "r") as f:
        labels = json.load(f)
        # labels = [[label[1], label[2]] for label in labels]

    transform = transforms.Compose(
        [
            transforms.Resize((SIZE, SIZE)),
            transforms.ToTensor(),
        ]
    )

    images = []
    target = []

    for i, filename in enumerate(os.listdir(PATH_PNG)):
        img = Image.open(os.path.join(PATH_PNG, filename))
        img = transform(img)
        images.append(img)
        target.append(labels[i])
        print(f"Loading image {i+1: 4d}/{len(os.listdir(PATH_PNG))}", end="\r")
    images_tensor = torch.stack(images)
    target_tensor = torch.tensor(target)
    dataset = TensorDataset(images_tensor, target_tensor)

    data_loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    print("\n")
    return data_loader


# %% [markdown]
# Display data


def affichage(data_loader):
    print("Taille du dataset :", len(data_loader.dataset))
    print("Taille du batch :", data_loader.batch_size)
    print("Nombre total de batches :", len(data_loader))

    # Afficher les caractéristiques de chaque batch
    for i, batch in enumerate(data_loader, 1):
        if i == 1:
            print("Caractéristiques du batch", i, ":")
            images, labels = batch
            print("Taille des images :", images.size())
            print("Taille des étiquettes :", labels.size())
    print("\n")


# %% [markdown]
# Encoder


class Encoder(nn.Module):

    def __init__(self, use_batchnorm=False, use_dropout=False):
        super(Encoder, self).__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.output_dim = LATENT_DIM
        self.layers = LAYERS
        self.kernels = KERNELS
        self.channels = CHANNELS
        self.strides = STRIDES
        self.padding = PADDING
        self.linear_dim = LINEAR_DIM
        self.conv = self.conv_block()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.linear_dim, self.output_dim)

    def conv_block(self):
        conv_layers = nn.Sequential()
        for i in range(self.layers):
            conv_layers.append(
                nn.Conv2d(
                    self.channels[i],
                    self.channels[i + 1],
                    self.kernels[i],
                    self.strides[i],
                    self.padding[i],
                )
            )
            if self.use_batchnorm:
                conv_layers.append(nn.BatchNorm2d(self.channels[i + 1]))
            conv_layers.append(nn.GELU())
            #
            if self.use_dropout:
                conv_layers.append(nn.Dropout2d(0.15))

        return conv_layers

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


# %% [markdown]
# Decoder
class Decoder(nn.Module):

    def __init__(self, use_batchnorm=False, use_dropout=False):
        super(Decoder, self).__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.input_dim = LATENT_DIM

        self.size = int((SIZE / (2**LAYERS)))
        self.layers = LAYERS
        self.kernels = KERNELS
        self.channels = CHANNELS[::-1]
        self.strides = STRIDES
        self.padding = PADDING
        self.linear_dim = LINEAR_DIM
        self.conv = self.conv_block()
        self.linear = nn.Linear(self.input_dim, self.linear_dim)
        self.output = nn.Conv2d(self.channels[-1], 3, kernel_size=1, stride=1)

    def conv_block(self):
        conv_layers = nn.Sequential()
        for i in range(self.layers):
            conv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.channels[i],
                    out_channels=self.channels[i + 1],
                    kernel_size=self.kernels[i],
                    stride=self.strides[i],
                    padding=self.padding[i],
                    output_padding=self.padding[i],
                )
            )
            if self.use_batchnorm and i != self.layers - 1:
                conv_layers.append(nn.BatchNorm2d(self.channels[i + 1]))

            conv_layers.append(nn.GELU())

            if self.use_dropout:
                conv_layers.append(nn.Dropout2d(0.15))
        return conv_layers

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(-1, CHANNELS[-1], self.size, self.size)
        x = self.conv(x)
        x = self.output(x)
        return x


# %% [markdown]
# Autoencoder
class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train(model, dataloader, criterion, optimizer, state="encoder"):
    model.train()
    train_loss = 0.0
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            if state == "encoder":
                a = 0
                b = 1
            elif state == "decoder":
                a = 1
                b = 0
            x = batch[a].to(DEVICE)
            y_recons = model(x)
            x = batch[b].to(DEVICE)
            loss = criterion(y_recons, x)
        train_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache()
        del x, y_recons
    train_loss /= len(dataloader)
    return train_loss


# %% [markdown]
# Validation function
def validate(model, dataloader, criterion, state):
    model.eval()
    valid_loss = 0.0
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            if state == "encoder":
                a = 0
                b = 1
            elif state == "decoder":
                a = 1
                b = 0
            x = batch[a].to(DEVICE)
            y_recons = model(x)
            x = batch[b].to(DEVICE)
            loss = criterion(y_recons, x)
        valid_loss += loss.item()
        torch.cuda.empty_cache()
        del x, y_recons
    valid_loss /= len(dataloader)
    return valid_loss


# %% [markdown]
# Affichage de l'image
def affichage_image(xs, ys):
    xs, ys = [float(xs)], [float(ys)]
    points = points = list(zip(xs, ys))
    coords = torch.tensor(points).unsqueeze(1).to(DEVICE)

    fig1, axes1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    autoencoder.eval()
    with torch.no_grad():
        generates = [autoencoder.decoder(coord) for coord in coords]

    im = generates[0].squeeze().detach().cpu().numpy().transpose(1, 2, 0)

    # Normalizing the image to be in the range [0, 1]
    im_normalized = (im - im.min()) / (im.max() - im.min())

    # Display the normalized image
    axes1.imshow(im_normalized, cmap="gray")
    axes1.axis("off")
    coord = coords[0].detach().cpu().numpy()[0]
    axes1.set_title(f"({coord[0]:.3f}, {coord[1]:.3f})")

    plt.show()


def plotting(model, data_loader):
    model.eval()
    labels = []
    for i, batch in enumerate(data_loader):
        x = batch[0].to(DEVICE)
        with torch.no_grad():
            y_recons = model.encoder(x)
            labels.append(y_recons.cpu().numpy().tolist())
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            x, y = labels[i][j]
            color = "green"
            plt.scatter(x, y, c=color, s=5, alpha=0.5)
            # print(f"Point {i*config['batch_size']+j+1}/{len(labels)*config['batch_size']}")
    ax.grid(True, color="lightgray", alpha=1.0, zorder=0)
    ax.set_title("Latent Space Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.savefig(f"{PATH}/frame.png")  # Sauvegarde de l'image avant l'affichage
    plt.show()  # Affichage de la figure à l'écran
    plt.close()
    model.train()


def onclick(event):
    x = event.xdata
    y = event.ydata
    affichage_image(x, y)


# %% [markdown]
# Parameters
config = {
    "batch_size": 32,
    "epochs": 20,
    "lr": 5e-4,
}

PATH = ".data\Mario_data"
PATH_TRAIN_JSON = "./.data/Mario_data/Train/labels/Mario_World.json"
PATH_TRAIN_PNG = "./.data/Mario_data/Train/screenshots/"
PATH_VALID_JSON = "./.data/Mario_data/Valid/labels/Mario_World.json"
PATH_VALID_PNG = "./.data/Mario_data/Valid/screenshots/"
# %% [markdown]
# Main


SIZE, LAYERS, LATENT_DIM, config["epochs"] = (
    int(input("Size of input images :")),
    int(input("Number of layers :")),
    int(input("Dimension of latent space :")),
    int(input("Number of epochs :")),
)


variables(SIZE, LAYERS, LATENT_DIM)
encoder = Encoder().to(DEVICE)
decoder = Decoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer_encoder = torch.optim.AdamW(
    encoder.parameters(), lr=config["lr"], weight_decay=1e-5
)
optimizer_decoder = torch.optim.AdamW(
    decoder.parameters(), lr=config["lr"], weight_decay=1e-5
)
scaler = torch.cuda.amp.GradScaler()
print(" Loading data...")
train_loader = load_data(PATH_TRAIN_JSON, PATH_TRAIN_PNG)
print("Training Data loaded")
valid_loader = load_data(PATH_VALID_JSON, PATH_VALID_PNG)
print("Validation Data loaded")
print("Training...")
print(f"Size: {SIZE} Layers: {LAYERS} Latent dim: {LATENT_DIM}")
print("\n")
for i in range(config["epochs"]):
    train_loss_encoder = train(
        encoder, train_loader, criterion, optimizer_encoder, state="encoder"
    )
    valid_loss_encoder = validate(encoder, valid_loader, criterion, state="encoder")
    train_loss_decoder = train(
        decoder, train_loader, criterion, optimizer_decoder, state="decoder"
    )
    valid_loss_decoder = validate(decoder, valid_loader, criterion, state="decoder")
    print(f"Epoch {i+1}/{config['epochs']}")
    print(
        f"Train loss: {train_loss_encoder:.4f} Validation loss: {valid_loss_encoder:.4f}"
    )
    print(
        f"Train loss: {train_loss_decoder:.4f} Validation loss: {valid_loss_decoder:.4f}"
    )
torch.save(
    encoder.state_dict(),
    f".data/Mario_data/Models/SIZE_{SIZE}_LAYERS_{LAYERS}_ENCODER.pth",
)
torch.save(
    decoder.state_dict(),
    f".data/Mario_data/Models/SIZE_{SIZE}_LAYERS_{LAYERS}_DECODER.pth",
)
autoencoder = AutoEncoder().to(DEVICE)
