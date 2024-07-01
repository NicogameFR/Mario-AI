# %%
import torch
from torch import nn
import torch.nn.functional as F

from torch.distributions import Normal

# %%


class SamplingLayer(nn.Module):
    def __init__(self, latent_dim):
        super(SamplingLayer, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, inputs):
        z_mean, z_log_var = inputs
        batch_size, latent_dim = z_mean.shape

        eps = Normal(0, 1).sample((batch_size, latent_dim)).to(z_mean.device)
        z = z_mean + torch.exp(z_log_var / 2) * eps

        return z


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.flatten = nn.Flatten()

        self.fc_mean = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_log_var = nn.Linear(256 * 16 * 16, latent_dim)

        self.sampling = SamplingLayer(latent_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x, indice1 = self.pool1(self.relu(self.bn2(self.conv2(x))))
        x, indice2 = self.pool2(self.relu(self.bn3(self.conv3(x))))
        x = self.relu(self.bn4(self.conv4(x)))

        x = self.flatten(x)

        z_mean = self.fc_mean(x)

        z_log_var = self.fc_log_var(x)

        z = self.sampling((z_mean, z_log_var))
        return z, z_mean, z_log_var, indice1, indice2


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
        self.conv_transpose2 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.conv_transpose3 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.transpose_maxpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv_transpose4 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.conv_transpose5 = nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, indice1, indice2):

        x = self.relu(self.fc(z))
        x = x.view(-1, 256, 16, 16)

        x = self.relu(self.conv_transpose2(x))
        x = self.relu(self.conv_transpose3(self.transpose_maxpool(x, indice2)))
        x = self.relu(self.conv_transpose4(self.transpose_maxpool(x, indice1)))
        x = self.sigmoid(self.conv_transpose5(x))
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        z, z_mean, z_log_var, indice1, indice2 = self.encoder(x)
        output = self.decoder(z, indice1, indice2)

        return x, z_mean, z_log_var, output
