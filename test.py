import torch
import os 



print("Loading model")
os.system("cls")
print("Locals models :")
for file in os.listdir(".data/Mario_data/Models/"):
    print(file)
loop2 = True
while loop2:
    model = input("Which model do you want to load? ")
    if model in os.listdir(".data/Mario_data/Models/"):
        loop2 = False
    else:
        print("This model does not exist, please try again")
# Spécifiez le chemin du fichier de sauvegarde
chemin_fichier = f".data/Mario_data/Models/{model}"

# Chargez le fichier de sauvegarde
informations_modele = torch.load(chemin_fichier, map_location=torch.device('cpu'))
# Accédez aux informations du modèle à partir du dictionnaire
# Par exemple, si vous voulez accéder au state_dict, vous pouvez faire :


# Vous pouvez maintenant inspecter ou manipuler state_dict selon vos besoins
# Par exemple, vous pouvez imprimer les clés et les valeurs de state_dict
for key, value in informations_modele.items():
    print(key)
