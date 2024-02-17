#%%
from torchvision import transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

#%%
with open('Mario_World.json', 'r') as f:
    labels = json.load(f)


transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Redimensionner les images en 64x64 pixels
    transforms.ToTensor()          # Convertir les images en tenseurs
])

# Créer des listes vides pour stocker les images et les étiquettes
images = []
target = []

# Parcourir le répertoire contenant les images .png
for filename in os.listdir('images'):
    if filename.endswith('.png'):
        # Charger l'image
        img = Image.open(os.path.join('images', filename))
        # Appliquer la transformation
        img = transform(img)
        # Ajouter l'image à la liste
        images.append(img)
        # Ajouter l'étiquette correspondante à partir du dictionnaire
        label = labels[filename]
        target.append(label)

# Convertir les listes en tenseurs
images_tensor = torch.stack(images)
target_tensor = torch.tensor(target)

# Créer un ensemble de données à partir des tenseurs d'images et d'étiquettes
dataset = TensorDataset(images_tensor, target_tensor)

# Créer un DataLoader pour charger les données en mini-lots
batch_size = 64
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Itérer sur les mini-lots de données
for images_batch, labels_batch in data_loader:
    # Ton code de traitement des mini-lots ici
    print("Taille du mini-lot d'images:", images_batch.size())
    print("Taille du mini-lot d'étiquettes:", labels_batch.size())
    break  # Interruption après le premier mini-lot pour cet exemple

#%%
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*8*8, 3) # Vecteur latent à 3 dimensions
        )
        
        # Décodeur
        self.decoder = nn.Sequential(
            nn.Linear(3, 128*8*8),
            nn.ReLU(),
            nn.Unflatten(-1, (128, 8, 8)), # Transformation du vecteur latent en volume 3D
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # Couche de convolution transposée avec stride de 2 et padding de 1
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # Couche de convolution transposée avec stride de 2 et padding de 1
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1), # Couche de convolution transposée avec stride de 2 et padding de 1
            nn.Sigmoid() # Pour obtenir des valeurs entre 0 et 1
        )
    
    def forward(self, x):
        # Passer par l'encodeur puis le décodeur
        latent_vector = self.encoder(x)
        reconstructed_output = self.decoder(latent_vector)
        return reconstructed_output, latent_vector

# Créer une instance de l'autoencodeur
autoencoder = Autoencoder()

# Exemple d'utilisation
input_data = torch.randn(1, 3, 64, 64) # Entrée de taille (batch_size, channels, height, width)
output_data, latent_vector = autoencoder(input_data) # Sortie de taille (batch_size, channels, height, width) et vecteur latent

# Convertir la sortie de l'autoencodeur en une image pouvant être affichée
output_image = output_data.squeeze().detach().numpy().transpose(1, 2, 0) # (channels, height, width) => (height, width, channels)

# Afficher l'image reconstruite
plt.imshow(output_image)
plt.axis('off')
plt.show()

# %%
