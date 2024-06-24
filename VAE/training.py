import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# importation des modules
from VAE import VAE
from VariationalLossLayer import VariationalLossLayer

# configuration du periférique de calcul (GPU ou CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# mise en place de la fonction d'entrainement
def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs.to(DEVICE))
            loss = criterion(outputs, inputs.to(DEVICE))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Print statistics
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}"
        )

    print("Finished Training")

# mise en place de la fonction de test
def test(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs.to(DEVICE))
            loss = criterion(outputs, inputs.to(DEVICE))
            running_loss += loss.item()

    print(f"Test Loss: {running_loss / len(dataloader)}")

# mise en place de la fonction de traitement des données
def collate_fn(batch):
    process = transforms.Compose([transforms.ToTensor()])

    # x - images; we process each image in the batch
    x = [process(data[0]) for data in batch]
    x = torch.concat(x).unsqueeze(1)

    # y - labels, note that we should convert the labels to LongTensor
    y = torch.LongTensor([data[1] for data in batch])

    return x, y



if __name__ == "__main__":

    # configuration
    config = {"batch_size": 64, "epochs": 10, "lr": 0.001, "latent_dim": 2}

    # téléchargement des données
    train_data = datasets.MNIST("../.data/", train=True, download=True)
    valid_data = datasets.MNIST("../.data/", train=False, download=True)

    # mettre en place les dataloaders
    train_loader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    # initialisation du modèle
    vae = VAE(config["latent_dim"]).to(DEVICE)
    # initialisation de la fonction de perte
    criterion = VariationalLossLayer()
    # initialisation de l'optimiseur
    optimizer = torch.optim.Adam(vae.parameters(), lr=config["lr"])
    # entrainement du modèle
    train(vae, train_loader, criterion, optimizer, config["epochs"])
    # test du modèle
    test(vae, valid_loader, criterion)
