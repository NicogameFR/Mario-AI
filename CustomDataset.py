# %%
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from matplotlib import pyplot as plt
from random import randint
import math

# %%


class CustomImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.Grayscale(),
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


def display_images_grid(dataset, num_images=9):

    fig, axes = plt.subplots(3, 3, figsize=(5, 5))
    images = []
    for i in range(num_images):
        x_img = randint(0, len(dataset))
        images.append(dataset[x_img][0])
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i-1].squeeze(0), cmap="gray")
        ax.axis("off")  # Cacher les axes
    plt.tight_layout()
    plt.show()


# %%
if __name__ == "__main__":
    dataset = CustomImageDataset(".data\Mario_data")
    display_images_grid(dataset)

# %%