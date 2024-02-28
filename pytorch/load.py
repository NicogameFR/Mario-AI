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


import time
import matplotlib.pyplot as plt

# Checking is CUDA available on current machine
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Project running on device: ", DEVICE)

# configurations for the task
config = {
    "batch_size": 64,
    "epochs": 10,
    "lr": 5e-4,  # learning rate
}


def collate_fn(batch):
    process = transforms.Compose([transforms.ToTensor(), transforms.Pad([2])])

    # x - images; we process each image in the batch
    x = [process(data[0]) for data in batch]
    x = torch.concat(x).unsqueeze(1)

    # y - labels, note that we should convert the labels to LongTensor
    y = torch.LongTensor([data[1] for data in batch])

    return x, y


# Paramètres du modèle :
LAYERS = 3
KERNELS = [3, 3, 3]
CHANNELS = [32, 64, 128]
STRIDES = [2, 2, 2]
LINEAR_DIM = 2048


class Encoder(nn.Module):

    def __init__(self, output_dim=2, use_batchnorm=False, use_dropout=False):
        super(Encoder, self).__init__()

        # bottleneck dimentionality
        self.output_dim = output_dim

        # variables deciding if using dropout and batchnorm in model
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        # convolutional layer hyper parameters
        self.layers = LAYERS
        self.kernels = KERNELS
        self.channels = CHANNELS
        self.strides = STRIDES
        self.conv = self.get_convs()

        # layers for latent space projection
        self.fc_dim = LINEAR_DIM
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.fc_dim, self.output_dim)

    def get_convs(self):
        """
        generating convolutional layers based on model's
        hyper parameters
        """
        conv_layers = nn.Sequential()
        for i in range(self.layers):
            # The input channel of the first layer is 1
            if i == 0:
                conv_layers.append(
                    nn.Conv2d(
                        1,
                        self.channels[i],
                        kernel_size=self.kernels[i],
                        stride=self.strides[i],
                        padding=1,
                    )
                )

            else:
                conv_layers.append(
                    nn.Conv2d(
                        self.channels[i - 1],
                        self.channels[i],
                        kernel_size=self.kernels[i],
                        stride=self.strides[i],
                        padding=1,
                    )
                )

            if self.use_batchnorm:
                conv_layers.append(nn.BatchNorm2d(self.channels[i]))

            # Here we use GELU as activation function
            conv_layers.append(nn.GELU())

            if self.use_dropout:
                conv_layers.append(nn.Dropout2d(0.15))

        return conv_layers

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.linear(x)

enccoder = Encoder()
enccoder = enccoder.to(DEVICE)
#summary(enccoder, (1, 28, 28))

class Decoder(nn.Module):

    def __init__(self, input_dim=2, use_batchnorm=False, use_dropout=False):

        super(Decoder, self).__init__()

        # variables deciding if using dropout and batchnorm in model
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        self.fc_dim = LINEAR_DIM
        self.input_dim = input_dim

        # Conv layer hypyer parameters
        self.layers = LAYERS
        self.kernels = KERNELS
        self.channels = CHANNELS[::-1]  # flip the channel dimensions
        self.strides = STRIDES

        # In decoder, we first do fc project, then conv layers
        self.linear = nn.Linear(self.input_dim, self.fc_dim)
        self.conv = self.get_convs()

        self.output = nn.Conv2d(self.channels[-1], 1, kernel_size=1, stride=1)

    def get_convs(self):
        conv_layers = nn.Sequential()
        for i in range(self.layers):

            if i == 0:
                conv_layers.append(
                    nn.ConvTranspose2d(
                        self.channels[i],
                        self.channels[i],
                        kernel_size=self.kernels[i],
                        stride=self.strides[i],
                        padding=1,
                        output_padding=1,
                    )
                )

            else:
                conv_layers.append(
                    nn.ConvTranspose2d(
                        self.channels[i - 1],
                        self.channels[i],
                        kernel_size=self.kernels[i],
                        stride=self.strides[i],
                        padding=1,
                        output_padding=1,
                    )
                )

            if self.use_batchnorm and i != self.layers - 1:
                conv_layers.append(nn.BatchNorm2d(self.channels[i]))

            conv_layers.append(nn.GELU())

            if self.use_dropout:
                conv_layers.append(nn.Dropout2d(0.15))

        return conv_layers

    def forward(self, x):
        x = self.linear(x)
        # reshape 3D tensor to 4D tensor
        x = x.reshape(x.shape[0], 128, 4, 4)
        x = self.conv(x)
        return self.output(x)

decoder = Decoder()
decoder = decoder.to(DEVICE)
#summary(decoder, (2,))

class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(output_dim=2, use_batchnorm=True, use_dropout=True)
        self.decoder = Decoder(input_dim=2, use_batchnorm=True, use_dropout=True)

    def forward(self, x):
        return self.decoder(self.encoder(x))

autoencoder = AutoEncoder()
autoencoder = autoencoder.to(DEVICE)
#summary(autoencoder, (1, 28, 28))

def plotting(step: int = 0, show=False, train_loader=None):

    model.eval()  # Switch the model to evaluation mode

    points = []
    label_idcs = []

    path = "./ScatterPlots"
    if not os.path.exists(path):
        os.mkdir(path)

    for i, data in enumerate(train_loader):
        img, label = [d.to(DEVICE) for d in data]
        # We only need to encode the validation images
        proj = model.encoder(img)
        points.extend(proj.detach().cpu().numpy())
        label_idcs.extend(label.detach().cpu().numpy())
        del img, label

    points = np.array(points)

    # Creating a scatter plot
    fig, ax = plt.subplots(figsize=(10, 10) if not show else (8, 8))
    scatter = ax.scatter(
        x=points[:, 0],
        y=points[:, 1],
        s=2.0,
        c=label_idcs,
        cmap="tab10",
        alpha=0.9,
        zorder=2,
    )

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
        plt.savefig(f"{path}/Step_{step:03d}.png", bbox_inches="tight")
        plt.close()  # don't forget to close the plot, or it is always in memory
        model.train()


def onclick(event):
    x = event.xdata
    y = event.ydata
    affichage_image(x, y)


def affichage_image(xs, ys):
    xs, ys = [float(xs)], [float(ys)]
    points = points = list(zip(xs, ys))
    coords = torch.tensor(points).unsqueeze(1).to(DEVICE)

    fig1, axes1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    model.eval()
    with torch.no_grad():
        generates = [model.decoder(coord) for coord in coords]

    # plot points
    im = generates[0].squeeze().detach().cpu()
    axes1.imshow(im, cmap="gray")
    axes1.axis("off")
    coord = coords[0].detach().cpu().numpy()[0]
    axes1.set_title(f"({coord[0]:.3f}, {coord[1]:.3f})")

    plt.show()


model = AutoEncoder()
model = model.to(DEVICE)


def start(train_data,valid_data):
# put datasets into dataloaders
    train_loader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)

    # For mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    steps = 0  # tracking the training steps
    plotting(0, show=True, train_loader=train_loader)


while True:
    input_model = input("Choose the model to load : 1 or 2 ?\n ----->>>   ")
    if input_model == "1":
        print('Processing .\r', end='')
        time.sleep(0.3)
        print('Processing ..\r', end='')
        time.sleep(0.3)
        print('Processing ...\r', end='')
        time.sleep(0.3)     
        model.load_state_dict(torch.load("autoencoder.pth"))
        train_data = datasets.MNIST(".data", train=True, download=True)
        valid_data = datasets.MNIST(".data", train=False, download=True)
        try:
            start(train_data,valid_data)
        except :
            continue
    elif input_model == "2": 
        model.load_state_dict(torch.load("autoencoder2.pth"))
        train_data = datasets.FashionMNIST(".data", train=True, download=True)
        valid_data = datasets.FashionMNIST(".data", train=False, download=True)
        try:
            start(train_data,valid_data)
        except :
            continue
    else:
        print("Invalid input\n")