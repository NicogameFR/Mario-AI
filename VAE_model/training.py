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
def train(model, dataloader, criterion, optimizer, pbar):
    model.train()
    running_loss = 0.0
    num_batches = 1
    for inputs in dataloader:
        inputs = inputs[0].to(DEVICE)
        # if torch.any(torch.isnan(inputs)):
        #   print(f"Skipping batch {num_batches}/{len(dataloader)}\n")
        #   num_batches += 1
        #   continue

        optimizer.zero_grad()
        outputs = model(inputs)

        loss, r_loss, kl_loss = criterion(outputs)
        num_batches += 1
        loss.backward()
        optimizer.step()
        running_loss = running_loss + loss.item()
        pbar.update(1)
    pbar.set_postfix(
        loss=f"{running_loss/num_batches:.4f}",
    )
    return running_loss


# mise en place de la fonction de traitement des données


transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
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
        )  # Assuming batch_size = 16, adjust according to your batch size
        print(batch_images[i].shape)
        plt.imshow(batch_images[i][0])
        plt.title(batch_filenames[i])
        plt.axis("off")

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


# %%

config = {
    "batch_size": 32,
    "epochs": 10,
    "lr": 0.001,
    "latent_dim": 8,
}

PNG_PATH = "..\.data\Mario_data"
dataset = CustomImageDataset(PNG_PATH, transform=transform)
dataloader = DataLoader(
    dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
)
# %%
# Initialiser les accumulateurs pour les sommes et les carrés des sommes
# mean = 0.
# std = 0.
# nb_samples = 0.
#
# for data in dataloader:
#    data = data[0]
#    batch_samples = data.size(0)
#    data = data.view(batch_samples, data.size(1), -1)
#    mean += data.mean(2).sum(0)
#    std += data.std(2).sum(0)
#    nb_samples += batch_samples
#
# mean /= nb_samples
# std /= nb_samples
#
# print(f'Mean: {mean}')
# print(f'Std: {std}')
#
# affichage_image(dataloader)
# %%

vae = VAE(config["latent_dim"]).to(DEVICE)

criterion = VariationalLossLayer()

optimizer = torch.optim.Adam(vae.parameters(), lr=config["lr"])

# %%


for i in range(config["epochs"]):
    with tqdm.tqdm(
        total=len(dataloader), desc="Entrainement", unit="Batch", ncols=75
    ) as pbar:
        train(vae, dataloader, criterion, optimizer, pbar)
