# This file contains the implementation of the Variational Autoencoder (VAE) model.
import torch
from torch import nn
import torch.nn.functional as F


from SamplingLayer import SamplingLayer


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_log_var = nn.Linear(64 * 7 * 7, latent_dim)
        self.sampling = SamplingLayer(latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.flatten(x)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)

        z = self.sampling((z_mean, z_log_var))
        return z, z_mean, z_log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.transpose_conv1 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
        self.transpose_conv2 = nn.ConvTranspose2d(
            64, 64, 3, stride=2, padding=1, output_padding=1
        )
        self.transpose_conv3 = nn.ConvTranspose2d(
            64, 32, 3, stride=2, padding=1, output_padding=1
        )
        self.transpose_conv4 = nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.transpose_conv1(x))
        x = F.relu(self.transpose_conv2(x))
        x = F.relu(self.transpose_conv3(x))
        x = torch.sigmoid(self.transpose_conv4(x))

        return x


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z, z_mean, z_log_var = self.encoder(x)
        output = self.decoder(z)

        return output, z_mean, z_log_var
