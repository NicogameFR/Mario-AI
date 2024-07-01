# %%
import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import pyscreenshot
import time

from VAE_model.module.VAE import VAE

# %%

# Détection de l'appareil
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du modèle
model = VAE().to(device)
model.load_state_dict(torch.load("VAE_model/VAE_model"))
model.eval()

# Transformation de l'image
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

# %%
# Boucle pour capturer et traiter les images
while True:
    time.sleep(0.1)

    # Capture d'écran
    screenshot = pyscreenshot.grab(bbox=(1167, 66, 1712, 483))

    # Conversion en image PIL puis application des transformations
    data = transform(screenshot).unsqueeze(0).to(device)

    # Passage à travers le modèle
    with torch.no_grad():
        recon_x, mu, logvar = model(data)

    # Conversion du tenseur en image pour affichage
    recon_x = recon_x[0].cpu().permute(1, 2, 0).numpy()

    # Affichage de l'image reconstruite
    plt.imshow(recon_x)
    plt.show()
