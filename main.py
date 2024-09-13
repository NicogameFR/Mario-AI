# %%
import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import pyscreenshot

from VAE import VAE

# %%

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(512, 1).to(DEVICE)
model.load_state_dict(torch.load("models\VAE_512_gray", weights_only=True))
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
)

# %%
while True:
    screenshot = pyscreenshot.grab(bbox=(1167, 66, 1712, 483))
    data = transform(screenshot).unsqueeze(0).to(DEVICE)
    img_recon = model(data)[0]
    img_recon = transforms.ToPILImage()(img_recon.squeeze(0))
    plt.imshow(img_recon, cmap="gray")
    plt.show()
