# %%
# modules pytoch
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# importation des modules personnalisés
# from module.VAE import VAE
from module.VariationalLossLayer import VariationalLossLayer
from module.VAE import VAE

# Autres
import os
from PIL import Image
from matplotlib import pyplot as plt
from math import sqrt
import numpy as np
import tqdm

# configuration du periférique de calcul (GPU ou CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# %%
# mise en place de la fonction d'entrainement
def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    num_batches = 1
    for inputs in dataloader:
        print(f"Batch {num_batches}/{len(dataloader)}\r", end='')
        inputs = inputs[0].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss, r_loss, kl_loss = criterion(outputs)
        num_batches += 1
        loss.backward()
        optimizer.step()
        running_loss = running_loss + loss.item()
        adv_loss = running_loss / num_batches
    return adv_loss


# mise en place de la fonction de traitement des données


transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
)


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = [
            img_name for img_name in os.listdir(img_dir) if img_name.endswith(".png")
        ]
        self.img_paths = [
            os.path.join(self.img_dir, img_name) for img_name in self.img_labels
        ]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels[idx]
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        return image, img_name


def affichage_image(dataloader):
    batch_images, batch_filenames = next(iter(dataloader))

    # Plot the images in a grid
    plt.figure(figsize=(10, 10))
    for i in range(len(batch_images)):
        plt.subplot(
            int(sqrt(config["batch_size"])), int(sqrt(config["batch_size"])), i + 1
        )
        plt.imshow(batch_images[i][0], cmap="gray")
        plt.title(batch_filenames[i])
        plt.axis("off")

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


# %%

config = {
    "batch_size": 64,
    "epochs": 5,
    "lr": 0.001,
    "latent_dim": 16,
}
PNG_PATH = "..\.data\Mario_data"
# image en couleur de 128*128 pixels
try:
    dataset = CustomImageDataset(PNG_PATH, transform=transform)

except:
    PNG_PATH = ".data\\Mario_data"
    dataset = CustomImageDataset(PNG_PATH, transform=transform)
dataloader = DataLoader(
    dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
)

affichage_image(dataloader)

# %%

vae = VAE(config["latent_dim"]).to(DEVICE)

criterion = VariationalLossLayer()

optimizer = torch.optim.Adam(vae.parameters(), lr=config["lr"])

# %%


r_loss = []
criterion = VariationalLossLayer(loss_weights=[1, 0.05])
vae = VAE(config["latent_dim"]).to(DEVICE)
for epoch in range(config["epochs"]):
    loss = train(vae, dataloader, criterion, optimizer)
    r_loss.append(loss)
    print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {loss}")
# %%


x = dataloader.dataset[0][0].unsqueeze(0).to(DEVICE)
x, z_mean, z_log_var, output = vae(x)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(x[0][0].cpu().detach().numpy(), cmap="gray")
plt.title("Original Image")
plt.axis("off")
# %%
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(output[0][0].cpu().detach().numpy(), cmap="gray")
plt.title("Reconstructed Image")
plt.axis("off")

# %%
