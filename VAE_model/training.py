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
# configuration du periférique de calcul (GPU ou CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# %%
# mise en place de la fonction d'entrainement
def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches = 1
        for inputs in dataloader:
            inputs = inputs[0].to(DEVICE)
            if torch.any(torch.isnan(inputs)):
                print(f"Skipping batch {num_batches}/{len(dataloader)}\n")
                num_batches += 1
                continue
            
            print(f"{num_batches}/{len(dataloader)}\r", end="")
            num_batches += 1
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs)

            loss.backward()
            optimizer.step()

            running_loss = running_loss + loss.item()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}"
        )

    print("Finished Training")


# mise en place de la fonction de traitement des données


transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
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
        return image, img_name  # Return both image and filename


def affichage_image(dataloader):
    batch_images, batch_filenames = next(iter(dataloader))

    # Plot the images in a grid
    plt.figure(figsize=(10, 10))
    for i in range(len(batch_images)):
        plt.subplot(
            int(sqrt(config["batch_size"])), int(sqrt(config["batch_size"])), i + 1
        )  # Assuming batch_size = 16, adjust according to your batch size
        plt.imshow(
            batch_images[i][0], cmap="gray"
        )  # Select the image tensor and use cmap='gray' for grayscale
        plt.title(batch_filenames[i])  # Display filename as title
        plt.axis("off")

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


# %%

config = {
    "batch_size": 64,
    "epochs": 100,
    "lr": 0.001,
    "latent_dim": 16,
}

PNG_PATH = "..\.data\Mario_data"
dataset = CustomImageDataset(PNG_PATH, transform=transform)
dataloader = DataLoader(
    dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
)
# %%
affichage_image(dataloader)
# %%

vae = VAE(config["latent_dim"]).to(DEVICE)

criterion = VariationalLossLayer()

optimizer = torch.optim.Adam(vae.parameters(), lr=config["lr"])

# %%

train(vae, dataloader, criterion, optimizer, config["epochs"])
