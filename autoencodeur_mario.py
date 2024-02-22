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

# %% [markdown]
# Parameters
config = {
    "batch_size": 32,
    "epochs": 100,
    "lr": 1e-4,
}

SIZE = 64

PATH = ".data\Mario_data"
PATH_TRAIN_JSON = "./.data/Mario_data/Train/labels/Mario_World.json"
PATH_TRAIN_PNG = "./.data/Mario_data/Train/screenshots/"
PATH_VALID_JSON = "./.data/Mario_data/Valid/labels/Mario_World.json"
PATH_VALID_PNG = "./.data/Mario_data/Valid/screenshots/"

LAYERS = 3
KERNELS = [3, 3, 3]
CHANNELS = [3, 32, 64, 128]
STRIDES = [2, 2, 2]
PADDING = [1, 1, 1]
LINEAR_DIM = 8 * 8 * 128
LATENT_DIM = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# %% [markdown]
# Load data


def load_data(PATH_JSON, PATH_PNG):
    with open(PATH_JSON, "r") as f:
        labels = json.load(f)
        labels = [[label[1], label[2]] for label in labels]

    transform = transforms.Compose(
        [
            transforms.Resize((SIZE, SIZE)),
            transforms.ToTensor(),
        ]
    )

    images = []
    target = []

    for i, filename in enumerate(os.listdir(PATH_PNG)):

        if filename.endswith(".png"):
            print(f"Chargement de {filename} {i+1: 4d}/{len(labels)}\r", end="")
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

    return data_loader


train_loader = load_data(PATH_TRAIN_JSON, PATH_TRAIN_PNG)
valid_loader = load_data(PATH_VALID_JSON, PATH_VALID_PNG)


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


affichage(train_loader)
affichage(valid_loader)

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

            if self.use_dropout:
                conv_layers.append(nn.Dropout2d(0.15))

        return conv_layers

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.linear(x)


encodeur = Encoder(use_batchnorm=True, use_dropout=True).to(DEVICE)


# %% [markdown]
# Decoder
class Decoder(nn.Module):

    def __init__(self, use_batchnorm=False, use_dropout=False):
        super(Decoder, self).__init__()
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.input_dim = LATENT_DIM

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
        x = x.reshape(-1, 128, 8, 8)
        x = self.conv(x)
        x = self.output(x)
        return x


decoder = Decoder(use_batchnorm=True, use_dropout=True).to(DEVICE)
# %% [markdown]
# Summary of encoder

# summary(encodeur, (3, 64, 64))
# %% [markdown]
# Summary of decoder

# summary(decoder, (2,))


# %% [markdown]
# Autoencoder
class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(use_batchnorm=True, use_dropout=True)
        self.decoder = Decoder(use_batchnorm=True, use_dropout=True)

    def forward(self, x):
        return self.decoder(self.encoder(x))


autoencoder = AutoEncoder().to(DEVICE)
# %% [markdown]
# Summary of autoencoder
summary(autoencoder, (3, 64, 64))

# %% [markdown]
# Training function
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    autoencoder.parameters(), lr=config["lr"], weight_decay=1e-5
)
scaler = torch.cuda.amp.GradScaler()


def train(model, dataloader, criterion, optimizer, save_distrib=False):
    model.train()
    train_loss = 0.0
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        x = batch[0].to(DEVICE)
        with torch.cuda.amp.autocast():
            x = batch[0].to(DEVICE)
            y_recons = model(x)
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
def validate(model, dataloader, criterion):
    model.eval()
    valid_loss = 0.0
    for i, batch in enumerate(dataloader):
        x = batch[0].to(DEVICE)
        with torch.no_grad():
            y_recons = model(x)
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

    # plot points
    im = generates[0].squeeze().detach().cpu()
    axes1.imshow(im, cmap="gray")
    axes1.axis("off")
    coord = coords[0].detach().cpu().numpy()[0]
    axes1.set_title(f"({coord[0]:.3f}, {coord[1]:.3f})")

    plt.show()


def plotting(model=autoencoder, show=False):
    model.eval()
    points = []
    label_idcs = []
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    for i, data in enumerate(train_loader):
        img, label = [d.to(DEVICE) for d in data]
        # We only need to encode the validation images
        proj = model.encoder(img)
        points.extend(proj.detach().cpu().numpy())
        label_idcs.extend(label.detach().cpu().numpy())
        del img, label
    print("Points shape:", np.array(points).shape)
    points = np.array(points)
    
    fig, ax = plt.subplots(figsize=(10, 10) if not show else (8, 8))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if show:
        ax.grid(True, color="lightgray", alpha=1.0, zorder=0)
        ax.set_title("Latent Space Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()
    else:
        # Do not show but only save the plot in training
        plt.axis("off")
        plt.savefig(f"{PATH}/frame.png", bbox_inches="tight")
        plt.close()  # don't forget to close the plot, or it is always in memory
        model.train()


def onclick(event):
    x = event.xdata
    y = event.ydata
    affichage_image(x, y)


# %% [markdown]
# Training loop
loop = True
while loop:
    loop = False
    ask = input("Do you want to train the model? (y/n) ")
    if ask == "y":
        for i in range(config["epochs"]):
            curr_lr = float(optimizer.param_groups[0]["lr"])
            train_loss = train(autoencoder, train_loader, criterion, optimizer)
            valid_loss = validate(autoencoder, valid_loader, criterion)
            print(
                f"Epoch {i+1}/{config['epochs']} Train loss: {train_loss:.4f} Validation loss: {valid_loss:.4f}\r",
                end="",
            )

        torch.save(
            autoencoder.state_dict(), f"./.data/mario_data/encoder_{valid_loss:.4f}.pth"
        )
    elif ask == "n":
        autoencoder.load_state_dict(torch.load(".data\Mario_data\encoder_0.0107.pth"))
        plotting(autoencoder, show=True)
    else:
        print("Invalid input")

# %%
