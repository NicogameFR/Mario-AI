import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalLossLayer(nn.Module):
    def __init__(self, loss_weights=[1, 0.06]):
        super(VariationalLossLayer, self).__init__()
        self.k1 = loss_weights[0]
        self.k2 = loss_weights[1]

    def forward(self, inputs):
        k1 = self.k1
        k2 = self.k2

        x, z_mean, z_log_var, y = inputs

        
        
        r_loss = F.binary_cross_entropy(y, x, reduction="sum")

        kl_loss = -torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        loss = k1 * r_loss + k2 * kl_loss

        return loss
