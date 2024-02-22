import torch
import numpy as np


a = [1,2,3].numpy()

b = a.transpose(1, 2, 0)

print(b)