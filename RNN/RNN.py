import torch
import torch.nn as nn
from RNN.CustomImageDataset import CustomImageDataset

training_data = CustomImageDataset("RNN/data")



rnn = nn.LSTM(128, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))