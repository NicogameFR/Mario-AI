#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np

from VAE import VAE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

