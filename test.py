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


def display_images_grid(dataset, avec_model=True, images = [], stop = False):

    fig, axes = plt.subplots(2, 3, figsize=(6, 4))
    if images == []:
        for i in range(6):
            x_img = randint(0, len(dataset))
            images.append(dataset[x_img][0])
    for i, ax in enumerate(axes.flat):
        if avec_model == True:
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = VAE().to(DEVICE)
            model.load_state_dict(
                torch.load("models\VAE", weights_only=True)
            )
            model.eval()
            img = images[i - 1]
            img_recon = model(img.unsqueeze(0).to(DEVICE))[0]
            img_recon= transforms.ToPILImage()(img_recon.squeeze(0))
            ax.imshow(img_recon, cmap="gray")
        else:
            ax.imshow(images[i - 1].squeeze(0), cmap="gray")
        ax.axis("off")  # Cacher les axes

    plt.tight_layout()
    plt.show()
    if stop == True:
        return
    display_images_grid(dataset, False, images, stop = True)

#%%


# %%
if __name__ == "__main__":
    dataset = CustomImageDataset(".data\Mario_data")
    #display_images_grid(dataset)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(DEVICE)
    #summary(model, (1, 128, 128))
    display_images_grid(dataset)
# %%
trois = torch.randn(3, 1, 128, 128)
