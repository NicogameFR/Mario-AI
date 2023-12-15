import pyscreenshot
# Charger l'image depuis le fichier
image = pyscreenshot.grab(childprocess=False, bbox = (1515, 54, 2324, 673))

image.show()
