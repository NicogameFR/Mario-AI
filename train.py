# %%
import torch
from torch import optim
from torch.utils.data import DataLoader
from VAE import VAE, vae_loss
from CustomDataset import CustomImageDataset
import datetime

#%%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(latent_dim=128).to(DEVICE)
dataset = CustomImageDataset(".data\Mario_data")
train_distrib , val_distrib = int((len(dataset) * 0.8)//1), int((len(dataset) * 0.2)//1)
if train_distrib+val_distrib != len(dataset):
    train_distrib += len(dataset)-(train_distrib+val_distrib)
train_set, val_set = torch.utils.data.random_split(
    dataset, [train_distrib , val_distrib ]
)

train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#%%
# Exemple d'entraînement
def train(model, optimizer, num_epochs=10, dataloader=train_dataloader):
    model.train()
    for epoch in range(num_epochs):
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
        val(model)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss / len(dataloader.dataset)}\n"
        )


def val(model, dataloader=val_dataloader):
    model.eval()
    val_loss = 0
    for batch_idx, data in enumerate(dataloader):
        data = data[0].to(DEVICE)
        with torch.no_grad():
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            val_loss += loss.item()

    print(f"Loss: {val_loss / len(dataloader.dataset)}")


train(model, optimizer, num_epochs=100, dataloader=train_dataloader)
# %%
torch.save(model.state_dict(), f"./models/VAE")
# %%
