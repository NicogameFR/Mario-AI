import torch
import torch.nn as nn
from CustomDataset import CustomImageDataset

training_data = CustomImageDataset("RNN/data")


rnn = nn.LSTM(128, 20, 4, device="cuda")
