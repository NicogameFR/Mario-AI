# %%
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from matplotlib import pyplot as plt
from random import randint
import torch
from VAE import VAE
from torchsummary import summary


# %%
class CustomImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.Grayscale(num_output_channels=1),
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


def display_images_grid(dataset,images, model):

    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i <= 4:
            img = images[i]
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            img_recon = model(img.unsqueeze(0).to(DEVICE))[0]
            img_recon = transforms.ToPILImage()(img_recon.squeeze(0))
            gif = img_recon
            ax.imshow(img_recon, cmap="gray")
            
            ax.axis("off")
        else:
            ax.imshow(images[i - 5].squeeze(0), cmap="gray")
            ax.axis("off")
    plt.tight_layout()
    plt.show()
    return gif




# %%
if __name__ == "__main__":
    dataset = CustomImageDataset(".data\Mario_data")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(1024, 1).to(DEVICE)
    model.load_state_dict(torch.load("models\VAE_1024_gray", weights_only=True))
    model.eval()
# %%
if __name__ == "__main__":
    images = []
    for i in range(5):
        images.append(dataset[randint(0, len(dataset))][0])
    display_images_grid(dataset, images, model)
# %%
