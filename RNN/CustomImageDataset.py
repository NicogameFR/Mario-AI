# %%
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import os

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


# Exemple d'entraînement
#def train(model, dataloader, optimizer, num_epochs=10):
#    model.train()
#    for epoch in range(num_epochs):
#        train_loss = 0
#        for batch_idx, data in enumerate(dataloader):
#            data = data[0].to(device)
#            optimizer.zero_grad()
#            recon_batch, mu, logvar = model(data)
#            loss = vae_loss(recon_batch, data, mu, logvar)
#            loss.backward()
#            train_loss += loss.item()
#            optimizer.step()
#
#        print(
#            f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss / len(dataloader.dataset)}"
#        )


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
