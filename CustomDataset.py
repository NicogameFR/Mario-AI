# %%
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from PIL import Image
import os
from VAE import VAE, vae_loss

# %%


class CustomImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )
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


# %%

if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=128).to(DEVICE)
    dataset = CustomImageDataset(".data\Mario_data")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Exemple d'entraînement
    def train(model, dataloader, optimizer, num_epochs=10):
        model.train()
        for epoch in range(num_epochs):
            train_loss = 0
            for batch_idx, data in enumerate(dataloader):
                print(f"batch {batch_idx}/{len(dataloader)}\r", end="")
                data = data[0].to(DEVICE)
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(data)
                loss = vae_loss(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss / len(dataloader.dataset)}"
            )

    train(model, dataloader, optimizer, num_epochs=100)