import cv2
import os


dossier_images = 'C:/Users/bapti/Desktop/IA/database/train/images/'

# Dossier contenant les fichiers texte
dossier_texte = 'C:/Users/bapti/Desktop/IA/database/train/labels/'

# Dossier de sortie pour les images modifiées
dossier_sortie = 'C:/Users/bapti/Desktop/IA/database/train/'

# Parcourir chaque fichier image dans le dossier
for nom_image in os.listdir(dossier_images):
    # Vérifier si le fichier est une image (extension jpg, png, etc.)
    if nom_image.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        chemin_image = os.path.join(dossier_images, nom_image)
        
        # Charger l'image
        image = cv2.imread(chemin_image)
        
        # Charger les coordonnées normalisées depuis le fichier texte correspondant
        nom_fichier_texte = os.path.splitext(nom_image)[0] + '.txt'
        chemin_fichier_texte = os.path.join(dossier_texte, nom_fichier_texte)
        
        if not os.path.exists(chemin_fichier_texte):
            print(f"Le fichier texte {nom_fichier_texte} n'existe pas pour l'image {nom_image}.")
            continue
        
        with open(chemin_fichier_texte, 'r') as fichier_texte:
            lignes = fichier_texte.readlines()

            # Extraire les coordonnées normalisées depuis le fichier texte
            coordonnees = lignes[0].strip().split()
            
            x_normalise = float(coordonnees[1])
            y_normalise = float(coordonnees[2])
            largeur_normalisee = float(coordonnees[3])
            hauteur_normalisee = float(coordonnees[4])

            # Dimensions de l'image
            hauteur, largeur, _ = image.shape

            # Calculer les coordonnées absolues du rectangle
            x_absolu = int((x_normalise - largeur_normalisee / 2) * largeur)
            y_absolu = int((y_normalise - hauteur_normalisee / 2) * hauteur)
            x2_absolu = int((x_normalise + largeur_normalisee / 2) * largeur)
            y2_absolu = int((y_normalise + hauteur_normalisee / 2) * hauteur)

            # Dessiner le rectangle sur l'image
            cv2.rectangle(image, (x_absolu, y_absolu), (x2_absolu, y2_absolu), (0, 255, 0), 2)

            # Sauvegarder l'image modifiée dans le dossier de sortie
            chemin_image_sortie = os.path.join(dossier_sortie, nom_image)
            cv2.imwrite(chemin_image_sortie, image)

# Afficher un message lorsque le traitement est terminé
print("Le traitement des images est terminé.")


