# %%
import torch
import numpy as np
import torch.nn as nn
import torchsummary
import torch.nn.functional as F
from torchvision import transforms
from torch.distributions.normal import Normal
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
import json
from PIL import Image
import os
import matplotlib.pyplot as plt
import time


DEVICE = "cpu"
# %%


# %%
class Encoder(nn.Module):

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 31 * 31, 16)
        self.mean_layer = nn.Linear(16, latent_dim)
        self.log_var_layer = nn.Linear(16, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, 64 * 31 * 31)
        x = F.relu(self.fc1(x))

        z_mean = self.mean_layer(x)
        z_log_var = self.log_var_layer(x)

        batch_size, _ = z_mean.shape

        epsilon = Normal(0, 1).sample((batch_size, latent_dim)).to(z_mean.device)

        z = z_mean + torch.exp(0.5 * z_log_var) * epsilon

        return z, z_mean, z_log_var


# %%


class Decoder(nn.Module):

    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64 * 31 * 31)
        self.conv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 64, 31, 31)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x


# %%


class VAE(nn.Module):

    def __init__(self, latent_dim, loss_weights, lr=1e-3):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.loss_weights = loss_weights
        self.optimizer = torch.optim.Adam(self.parameters(), lr)

    def forward(self, x):
        z, z_mean, z_log_var = self.encoder(x)

        x_reconstructed = self.decoder(z)

        return x_reconstructed

    def train_step(self, input):
        """
        Implementation of the training update.
        Receive an input, compute loss, get gradient, update weights and return metrics.
        Here, our metrics are loss.
        args:
            inputs : Model inputs
        return:
            loss    : Total loss
            r_loss  : Reconstruction loss
            kl_loss : KL loss
        """

        # ---- Get the input we need, specified in the .fit()
        #
        if isinstance(input, tuple):
            input = input[0]

        k1, k2 = self.loss_weights

        # ---- Reset grad
        #
        self.zero_grad()

        # ---- Forward pass
        #
        # Get encoder outputs
        #
        z_mean, z_log_var, z = self.encoder(input)

        # ---- Get reconstruction from decoder
        #
        reconstruction = self.decoder(z)

        # ---- Compute loss
        #      Total loss = Reconstruction loss + KL loss
        #
        r_loss = torch.nn.functional.binary_cross_entropy(
            reconstruction, input, reduction="sum"
        )
        kl_loss = -torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        loss = r_loss * k1 + kl_loss * k2

        # ---- Compute gradients for the weights
        #
        loss.backward()

        # ---- Adjust learning weights
        #
        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # ---- Update metrics (includes the metric that tracks the loss)
        #
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(input, reconstruction)

        # ---- Return a dict mapping metric names to current value
        # Note that it will include the loss (tracked in self.metrics).
        #
        return {m.name: m.result() for m in self.metrics}


# %%
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
    ]
)
taille = 500
output_size = 2


def load_data(PATH_PNG, transform=transform):
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

    images_tensor = torch.stack(images)

    dataset = TensorDataset(images_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def affichage(data_loader):
    print("Taille du dataset :", len(data_loader.dataset))
    print("Taille du batch :", data_loader.batch_size)
    print("Nombre total de batches :", len(data_loader))
    for i, batch in enumerate(data_loader, 1):
        print("Taille des images :", batch[0].size())
        break
    print("\n")


# %%
batch_size = 32
latent_dim = 2
input_dim = 128

train_loader, validation_loader = load_data(
    PATH_PNG="..\.data\Mario_data\Train\screenshots",
    transform=transform,
)
# %%

affichage(train_loader)
# %%

vae = VAE(latent_dim, [1, 0.06], 1e-3).to(DEVICE)
torchsummary.summary(vae, (3, input_dim, input_dim))

# %%

for _, data in enumerate(train_loader):
    data = data[0].to(DEVICE)
    loss = vae.train_step(data)
    print(loss)
# %%
