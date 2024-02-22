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
    print("\n")

    images_tensor = torch.stack(images)
    target_tensor = torch.tensor(target)
    dataset = TensorDataset(images_tensor, target_tensor)

    data_loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False, drop_last=True
    )
    os.system("cls")
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
        self.encoder = Encoder(use_batchnorm=True, use_dropout=True)
        self.decoder = Decoder(use_batchnorm=True, use_dropout=True)
        
    def forward(self, x):
        return self.decoder(self.encoder(x))


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
            plt.scatter(x, y, c=color, s=10, alpha=0.5)
            os.system("cls")
            print(f"Point {i*config['batch_size']+j+1}/{len(labels)*config['batch_size']}")
    os.system("cls")
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
# Main loop
loop1 = True
loaded = False
while loop1:
    ask = "y"
    ask = input("Do you want to load a local model ? (y/n) ")
    ask = ask.lower()
    ask = ask.strip()
    if ask == "n":
        loop1 = False
        print("Please put config variables of this model:")
        SIZE, LAYERS, LATENT_DIM, config["epochs"] = (
            int(input("Size of input images :")),
            int(input("Number of layers :")),
            int(input("Dimension of latent space :")),
            int(input("Number of epochs :")),
        )
        variables(SIZE, LAYERS, LATENT_DIM)
        autoencoder = AutoEncoder().to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            autoencoder.parameters(), lr=config["lr"], weight_decay=1e-5
        )
        scaler = torch.cuda.amp.GradScaler()
        if not loaded:
            train_loader = load_data(PATH_TRAIN_JSON, PATH_TRAIN_PNG)
            valid_loader = load_data(PATH_VALID_JSON, PATH_VALID_PNG)
            loaded = True
        for i in range(config["epochs"]):
            curr_lr = float(optimizer.param_groups[0]["lr"])
            train_loss = train(autoencoder, train_loader, criterion, optimizer)
            valid_loss = validate(autoencoder, valid_loader, criterion)
            print(
                f"Epoch {i+1}/{config['epochs']} Train loss: {train_loss:.4f} Validation loss: {valid_loss:.4f}"
            )
        torch.save(
            autoencoder.state_dict(),
            f".data/Mario_data/Models/SIZE_{SIZE}_LAYERS_{LAYERS}_LATENT_DIM_{LATENT_DIM}_LOSS_{valid_loss}.pth",
        )
    elif ask == "y":
        loop2 = True
        while loop2:
            print("Locals models :")
            for file in os.listdir(".data/Mario_data/Models/"):
                print(file)
            models = input("Which model do you want to load? ")
            if models in os.listdir(".data/Mario_data/Models/"):
                loop2 = False
                model = torch.load(f".data/Mario_data/Models/{models}")
                os.system("cls")
                print("Please put config variables of this model:")
                print(models)
                loop3 = True
                while loop3:
                    try:
                        SIZE, LAYERS, LATENT_DIM = (
                            int(input("Size: ")),
                            int(input("Layers: ")),
                            int(input("Latent dim: ")),
                        )
                        variables(SIZE, LAYERS, LATENT_DIM)
                        autoencoder = AutoEncoder().to(DEVICE)
                        sys.stdout = open('.data/Mario_data/output.txt', 'w')
                        summary(autoencoder, (3, SIZE, SIZE))
                        sys.stdout.close()
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.AdamW(
                            autoencoder.parameters(), lr=config["lr"], weight_decay=1e-5
                        )
                        scaler = torch.cuda.amp.GradScaler()
                        autoencoder.load_state_dict(model)
                        loop3 = False
                    except Exception as e:
                        print(e)
                        print("Invalid input")
                        print("Try again")
            else:
                os.system("cls")
                print("Invalid model")
        if not loaded:
            train_loader = load_data(PATH_TRAIN_JSON, PATH_TRAIN_PNG)
            valid_loader = load_data(PATH_VALID_JSON, PATH_VALID_PNG)
            loaded = True
        ask = input("Do you want to train the model ? (y/n) ")
        if ask == "y":
            loop4 = True
            while loop4:
                epochs = input("Epochs: ")
                try:
                    epochs = int(epochs)
                    if epochs > 0:
                        config["epochs"] = epochs
                        loop4 = False
                    else:
                        print("Invalid input")
                except:
                    print("Invalid input")
            for i in range(config["epochs"]):
                curr_lr = float(optimizer.param_groups[0]["lr"])
                train_loss = train(autoencoder, train_loader, criterion, optimizer)
                valid_loss = validate(autoencoder, valid_loader, criterion)
                print(
                    f"Epoch {i+1}/{config['epochs']} Train loss: {train_loss:.4f} Validation loss: {valid_loss:.4f}"
                )
            torch.save(
                autoencoder.state_dict(),
                f".data/Mario_data/Models/SIZE_{SIZE}_LAYERS_{LAYERS}_LATENT_DIM_{LATENT_DIM}_LOSS_{valid_loss}.pth",
            )
        elif ask == "n":
            os.system("cls")
            print("Model loaded")
            plotting(autoencoder, train_loader)
        else:
            print("Invalid input")
            loop1 = True

    else:
        print("Invalid input")
        loop1 = True
