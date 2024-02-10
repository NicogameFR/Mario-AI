# %%
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F


from torchvision import datasets, transforms

# %%
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)


def f(x, y):
    return x**2 + y**2


lr = 0.1
for _ in range(200):
    res = f(x, y)
    res.backward()
    with torch.no_grad():
        x -= lr * x.grad
        y -= lr * y.grad
    x.grad.zero_(), y.grad.zero_()

print(f"x = {x}, y = {y}, f(x, y) = {f(x, y)}")

# %%
transf = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.1307, 0.3087)]
)


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(".data/mnist", train=True, download=True, transform=transf),
    batch_size=32,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(".data/mnist", train=False, download=True, transform=transf),
    batch_size=32,
    shuffle=True,
)

for data_batch, label_batch in train_loader:
    fig = plt.figure()
    for i in range(1,31):
        data = data_batch[i][0]
        label = label_batch[i]
        plt.subplot(5,6,i)
        plt.axis('off')
        plt.title(label.item())
        plt.imshow(data, cmap='gray_r')
    
    break
plt.show()



# %%
