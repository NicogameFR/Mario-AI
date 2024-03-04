import random
from PIL import Image, ImageDraw
import json
import os

# Dimensions de l'image
width = 128
height = 128
data = []
# Création de l'image avec un fond blanc
for i in range(1000):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    # Position aléatoire de la croix rouge
    x = random.randint(0, width)
    y = random.randint(0, height)
    data.append([x, y])
    # Dessin de la croix rouge
    draw.line((x-4, y-4, x+4, y+4), fill="red", width=2)
    draw.line((x-4, y+4, x+4, y-4), fill="red", width=2)
    # Enregistrement de l'image
    image.save(f".data/gererate/{i}.png")

with open(".data/labels.json", "w") as f:
    json.dump(data, f)