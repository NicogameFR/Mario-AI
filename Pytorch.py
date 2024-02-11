# %%
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F


from torchvision import datasets, transforms


torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# x = torch.tensor(3.0, requires_grad=True)
# y = torch.tensor(3.0, requires_grad=True)
#
#
# def f(x, y):
#    return x**2 + y**2
#
#
# lr = 0.1
# for _ in range(200):
#    res = f(x, y)
#    res.backward()
#    with torch.no_grad():
#        x -= lr * x.grad
#        y -= lr * y.grad
#    x.grad.zero_(), y.grad.zero_()
#
# print(f"x = {x}, y = {y}, f(x, y) = {f(x, y)}")
#
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

# for data_batch, label_batch in train_loader:
#    fig = plt.figure()
#    for i in range(1, 31):
#        data = data_batch[i][0]
#        label = label_batch[i]
#        plt.subplot(5, 6, i)
#        plt.axis("off")
#        plt.title(label.item())
#        plt.imshow(data, cmap="gray_r")
#
#    break
# plt.show()


# %%
N_EPOCH = 13


class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 64)
        self.layer2 = nn.Linear(64, 10)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.ndim == 1:
                nn.init.normal_(p)
            else:
                nn.init.kaiming_normal_(p)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.layer1(x))
        x = F.softmax(self.layer2(x), dim=-1)
        return x


model = Dense()

lr = 1e-3

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def accuracy(proba_batch, label_batch):
    correct = 0
    batch_size = label_batch.size(0)
    preds = torch.argmax(proba_batch, dim=-1)
    for i, pred in enumerate(preds):
        if pred == label_batch[i]:
            correct += 1
    return correct / batch_size


for epoch in range(N_EPOCH):
    print(f"Epoch {epoch+1}/{N_EPOCH}")
    print
    for batch, (data_batch, label_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        pred_batch = model(data_batch)
        loss = criterion(pred_batch, label_batch)
        loss.backward()
        optimizer.step()
        acc = accuracy(pred_batch, label_batch)
        print(
            f"\r Batch = {batch+1}/1875  loss = {loss : .3}  acc = {acc : .3}",
            end="",
        )
    print("\n")
    loss, acc = 0., 0.
    batch = 0
    for data_batch, label_batch in train_loader:
        batch += 1
        with torch.no_grad():
            pred_batch = model(data_batch)
            loss += criterion(pred_batch, label_batch)
            acc += accuracy(pred_batch, label_batch)
    loss /= batch
    acc /= batch
    print(f"validation loss = {loss : .3} \nvalidation acc = {acc : .3}")

# %%

N_EPOCH = 13


x_axis , y_axis1 = list(range(1, N_EPOCH+1)) , []


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dense1 = nn.Linear(320, 50)
        self.dense2 = nn.Linear(50, 10)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.ndim == 1:
                nn.init.normal_(p)
            else:
                nn.init.kaiming_normal_(p)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.relu(F.dropout2d(x,p = 0.2 ,training=self.training))
        x = x.view(-1, 320)
        x = F.dropout(F.relu(self.dense1(x)),p = 0.2 , training=self.training)
        x = F.softmax(self.dense2(x), dim=-1)
        return x


model = CNN()

lr = 1e-3

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def accuracy(proba_batch, label_batch):
    correct = 0
    batch_size = label_batch.size(0)
    preds = torch.argmax(proba_batch, dim=-1)
    for i, pred in enumerate(preds):
        if pred == label_batch[i]:
            correct += 1
    return correct / batch_size


for epoch in range(N_EPOCH):
    model.training = True
    print(f"Epoch {epoch+1}/{N_EPOCH}")
    for batch, (data_batch, label_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        pred_batch = model(data_batch)
        loss = criterion(pred_batch, label_batch)
        loss.backward()
        optimizer.step()
        acc = accuracy(pred_batch, label_batch)
        print(
            f"\r Batch = {batch+1}/1875  loss = {loss : .3}  acc = {acc : .3}",
            end="",
        )
    print("\n")
    loss, acc = 0., 0.
    batch = 0
    model.training = False
    for data_batch, label_batch in train_loader:
        batch += 1
        with torch.no_grad():
            pred_batch = model(data_batch)
            loss += criterion(pred_batch, label_batch)
            acc += accuracy(pred_batch, label_batch)
    loss /= batch
    acc /= batch
    y_axis1.append(acc)
    print(f"validation loss = {loss : .3} \nvalidation acc = {acc : .3}")

fig = plt.figure()
ax = plt.gca()
ax.set_ylim(0.94,1.0)
plt.title("Validation accuracy")
plt.plot(x_axis, y_axis1, label="CNN")
plt.legend(loc = "upper left")

plt.show()

