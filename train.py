# %%
import torch
from torch import optim
from torch.utils.data import DataLoader
from VAE import VAE, vae_loss
from CustomDataset import CustomImageDataset, display_images_grid
from torchsummary import summary
import random


# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(1024, 1).to(DEVICE)
dataset = CustomImageDataset(".data\Mario_data")
train_distrib, val_distrib = int((len(dataset) * 0.8) // 1), int(
    (len(dataset) * 0.2) // 1
)
if train_distrib + val_distrib != len(dataset):
    train_distrib += len(dataset) - (train_distrib + val_distrib)
train_set, val_set = torch.utils.data.random_split(
    dataset, [train_distrib, val_distrib]
)

train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
images = []

gifs = []

#%%
for i in range(5):
    images.append(dataset[random.randint(0, len(dataset))][0])
print(
    f"Device: {DEVICE} - Train set: {len(train_dataloader.dataset)} - Val set: {len(val_dataloader.dataset)}"
)
# %%


def train(model, optimizer, dataloader=train_dataloader):
    model.train()
    epoch = 1
    while True:
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):
            print(f"batch {batch_idx+1}/{len(dataloader)}\r", end="")
            data = data[0].to(DEVICE)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(
            f"Epoch {epoch}, Loss: {train_loss / len(dataloader.dataset)}"
        )
        val(model)
        epoch += 1


def val(model, dataloader=val_dataloader):
    model.eval()
    val_loss = 0
    for batch_idx, data in enumerate(dataloader):
        print(f"batch {batch_idx+1}/{len(dataloader)}\r", end="")
        data = data[0].to(DEVICE)
        with torch.no_grad():
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            val_loss += loss.item()
    print(f"Validation Loss: {val_loss / len(dataloader.dataset)}\n")
    gifs.append(display_images_grid(dataloader.dataset, images, model))
    torch.save(model.state_dict(), f"./models/VAE")


# %%
train(model, optimizer, dataloader=train_dataloader)
# %%
save = gifs

new_size = (720, 720)
gifs = [gif.resize(new_size) for gif in gifs]


output_path = ".data/reconstitution/training.gif"
gifs[0].save(output_path, save_all=True, append_images=gifs[1:], loop=0, duration=50)

# %%
