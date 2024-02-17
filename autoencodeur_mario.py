# %%
from torchvision import transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader

# torchsummary for easy checking and debugging
from torchsummary import summary
from tqdm.notebook import tqdm
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Project running on device: ", DEVICE)

# configurations for the task
config = {
    "batch_size": 64,
    "epochs": 10,
    "lr": 5e-4,  # learning rate
}


# %%
with open("Mario_World.json", "r") as f:
    labels = json.load(f)


transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)

images = []
target = []

for i, filename in enumerate(os.listdir("./worlds")):
    if filename.endswith(".png"):
        img = Image.open(os.path.join("./worlds", filename))
        img = transform(img)
        images.append(img)
        label = labels[i]
        target.append(label)

images_tensor = torch.stack(images)
target_tensor = torch.tensor(target)
dataset = TensorDataset(images_tensor, target_tensor)
batch_size = 64
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# %%
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
            nn.Linear(128 * 8 * 8, 3),  # Vecteur latent à 3 dimensions
        )

        # Décodeur
        self.decoder = nn.Sequential(
            nn.Linear(3, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(-1, (128, 8, 8)),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent_vector = self.encoder(x)
        reconstructed_output = self.decoder(latent_vector)
        return reconstructed_output, latent_vector


autoencoder = Autoencoder()


input_data = torch.randn(1, 3, 64, 64)
output_data, latent_vector = autoencoder(input_data)


output_image = output_data.squeeze().detach().numpy().transpose(1, 2, 0)


plt.imshow(output_image)
plt.axis("off")
plt.show()

# %%

def plotting(step: int = 0, show=False):

    autoencoder.eval()  # Switch the model to evaluation mode

    points = []
    label_idcs = []

    path = "./ScatterPlots"
    if not os.path.exists(path):
        os.mkdir(path)

    for i, data in enumerate(data_loader):
        img, label = [d.to(DEVICE) for d in data]
        # We only need to encode the validation images
        proj = autoencoder.encoder(img)
        points.extend(proj.detach().cpu().numpy())
        label_idcs.extend(label.detach().cpu().numpy())
        del img, label

    points = np.array(points)

    # Creating a scatter plot
    fig, ax = plt.subplots(figsize=(10, 10) if not show else (8, 8))
    scatter = ax.scatter(
        x=points[:, 0],
        y=points[:, 1],
        s=2.0,
        c=label_idcs,
        cmap="tab10",
        alpha=0.9,
        zorder=2,
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if show:
        ax.grid(True, color="lightgray", alpha=1.0, zorder=0)
        ax.set_title("Latent Space Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    else:
        # Do not show but only save the plot in training
        plt.axis("off")
        plt.savefig(f"{path}/Step_{step:03d}.png", bbox_inches="tight")
        plt.close()  # don't forget to close the plot, or it is always in memory
        autoencoder.train()

def onclick(event):
    x = event.xdata
    y = event.ydata
    affichage_image(x,y)


# convert image sequence to a gif file
def save_gif():

    frames = []
    imgs = sorted(os.listdir("./ScatterPlots"))

    for im in imgs:
        new_frame = Image.open("./ScatterPlots/" + im)
        frames.append(new_frame)

    frames[0].save(
        "latentspace.gif",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=200,
        loop=0,
    )


# %%
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=config["lr"], weight_decay=1e-5)

# For mixed precision training
scaler = torch.cuda.amp.GradScaler()
steps = 0  # tracking the training steps


# %%
def train(model, dataloader, criterion, optimizer, save_distrib=False):
    # steps is used to track training progress, purely for latent space plots
    global steps

    model.train()
    train_loss = 0.0

    # Process tqdm bar, helpful for monitoring training process
    batch_bar = tqdm(
        total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="Train"
    )

    for i, batch in enumerate(dataloader):

        optimizer.zero_grad()
        x = batch[0].to(DEVICE)

        # Here we implement the mixed precision training
        with torch.cuda.amp.autocast():
            x = batch[0].to(DEVICE)
            y_recons = model(x)
            loss = criterion(y_recons, x)

        train_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_bar.set_postfix(
            loss=f"{train_loss/(i+1):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.4f}"
        )
        batch_bar.update()

        # Saving latent space plots
        # if steps % 10 == 0 and save_distrib and steps <= 400:
        #    plotting(steps)
        steps += 1

        # remove unnecessary cache in CUDA memory
        torch.cuda.empty_cache()
        del x, y_recons

    batch_bar.close()
    train_loss /= len(dataloader)

    return train_loss


# %%
def validate(model, dataloader, criterion):

    model.eval()  # Don't forget to turn the model to eval mode
    valid_loss = 0.0

    # Progress tqdm bar
    batch_bar = tqdm(
        total=len(dataloader),
        dynamic_ncols=True,
        leave=False,
        position=0,
        desc="Validation",
    )

    for i, batch in enumerate(dataloader):
        x = batch[0].to(DEVICE)

        with torch.no_grad():  # we don't need gradients in validation
            y_recons = model(x)
            loss = criterion(y_recons, x)

        valid_loss += loss.item()

        batch_bar.set_postfix(
            loss=f"{valid_loss/(i+1):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.4f}"
        )
        batch_bar.update()

        torch.cuda.empty_cache()
        del x, y_recons

    batch_bar.close()
    valid_loss /= len(dataloader)

    return valid_loss


# %%
for i in range(config["epochs"]):
    curr_lr = float(optimizer.param_groups[0]["lr"])
    train_loss = train(autoencoder, data_loader, criterion, optimizer, save_distrib=True)
    valid_loss = validate(autoencoder, data_loader, criterion)
    print(
        f"Epoch {i+1}/{config['epochs']}\nTrain loss: {train_loss:.4f}\t Validation loss: {valid_loss:.4f}\tlr: {curr_lr:.4f}"
    )
# %%
def affichage_image(xs, ys):
    xs , ys = [float(xs)], [float(ys)]
    points = points = list(zip(xs, ys))
    coords = torch.tensor(points).unsqueeze(1).to(DEVICE)

    fig1, axes1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    autoencoder.eval()
    with torch.no_grad():
        generates = [autoencoder.decoder(coord) for coord in coords]

    # plot points
    im = generates[0].squeeze().detach().cpu()
    axes1.imshow(im, cmap="gray")
    axes1.axis("off")
    coord = coords[0].detach().cpu().numpy()[0]
    axes1.set_title(f"({coord[0]:.3f}, {coord[1]:.3f})")

    plt.show()


plotting(450)
torch.save(autoencoder.state_dict(),'autoencoder_mario.pth')
plotting(500, show=True)
