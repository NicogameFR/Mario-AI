import torch
import torch.nn as nn
from torch.distributions import Normal

class SamplingLayer(nn.Module):
    def __init__(self, latent_dim):
        super(SamplingLayer, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = z_mean.size(0)

        eps = Normal(0, 1).sample((batch_size, self.latent_dim)).to(z_mean.device)
        z = z_mean + torch.exp(z_log_var / 2) * eps

        return z