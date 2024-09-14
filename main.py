# %%
import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import pyscreenshot

from VAE import VAE
from time import sleep

# %%

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(1024, 1).to(DEVICE)
model.load_state_dict(torch.load("models\VAE", weights_only=True))
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
)
gifs = []
# %%
while True:
    sleep(0.1)
    screenshot = pyscreenshot.grab(childprocess=False, bbox=(414, 47, 2149, 1370))
    data = transform(screenshot).unsqueeze(0).to(DEVICE)
    img_recon = model(data)[0]
    img_recon = transforms.ToPILImage()(img_recon.squeeze(0))
    gifs.append(img_recon)
    plt.imshow(img_recon, cmap="gray")
    plt.axis('off')
    plt.show()
    plt.close()

#%%
new_size = (1080, 1080)
gifs = [gif.resize(new_size) for gif in gifs]


output_path = ".data/reconstitution/main.gif"
gifs[0].save(output_path, save_all=True, append_images=gifs[1:], loop=0, duration=250)


# %%
