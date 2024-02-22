# %%

import pyscreenshot
import birdseyelib as bird
import os
import json
import random

# %%

HOST = "127.0.0.1"
PORT = 8080

PATH_TRAIN_JSON = "./.data/Mario_data/Train/labels/Mario_World.json"
PATH_TRAIN_PNG = "./.data/Mario_data/Train/screenshots/"
PATH_VALID_JSON = "./.data/Mario_data/Valid/labels/Mario_World.json"
PATH_VALID_PNG = "./.data/Mario_data/Valid/screenshots/"

TOTAL_FRAMES = 5000
SIZE = 640

#%%

class BizHawk:

    def __init__(self, HOST, PORT):
        self.client = bird.Client(HOST, PORT)
        self.memory = bird.Memory(self.client)
        self.controller_input = bird.ControllerInput(self.client)
        self.emulation = bird.Emulation(self.client)
        self.external_tool = bird.ExternalTool(self.client)
        self.client.connect()

    def __str__(self):
        return f"BizHawk Client: {self.client}"




bh = BizHawk(HOST, PORT)


# %%



if os.path.exists(PATH_TRAIN_JSON):
    os.remove(PATH_TRAIN_JSON)
if os.listdir(PATH_TRAIN_PNG):
    os.system(f"cd {PATH_TRAIN_PNG} & del /f *.png")
if os.path.exists(PATH_VALID_JSON):
    os.remove(PATH_VALID_JSON)
if os.listdir(PATH_VALID_PNG):
    print("Suppression des fichiers")
    os.system(f"cd {PATH_VALID_PNG} & del /f *.png")


# %%


def data_json(fichier_json, new_data):
    if os.path.exists(fichier_json):
        with open(fichier_json, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(new_data)
    with open(fichier_json, "w") as f:
        json.dump(data, f)


# %%

i = 0
print("Début de la capture")
bh.memory.add_address(0x04AC)  # Mario's x1 position
bh.memory.add_address(0x04AD)  # Mario's y1 position
bh.memory.add_address(0x04AE)  # Mario's x2 position
bh.memory.add_address(0x04AF)  # Mario's y2 position
bh.memory.add_address(0x071A)  # Décalage niveau
bh.memory.add_address(0x071C)  # Décalage écran
bh.memory.request_memory()
bh.client.advance_frame()

while bh.client.is_connected() and i < TOTAL_FRAMES:
    bh.memory.request_memory()
    x1 = bh.memory.get_memory()["0x4ac"]
    y1 = bh.memory.get_memory()["0x4ad"]
    x2 = bh.memory.get_memory()["0x4ae"]
    y2 = bh.memory.get_memory()["0x4af"]
    d1 = bh.memory.get_memory()["0x71a"]
    d2 = bh.memory.get_memory()["0x71c"]
    if [x1, y1, x2, y2] != [0, 0, 0, 0] or [x1, y1, x2, y2] != [-1, -1, -1, -1]:
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        decalage = d1 * 256 + d2
        print(
            f"Nombre de captures: {i+1}/{TOTAL_FRAMES} - Décalage: {decalage} - x: {x} - y: {y}\r",
            end="",
        )
        new_data = [decalage, x, y]
        image = pyscreenshot.grab(childprocess=False, bbox=(1167, 66, 1712, 483))
        #image = image.resize((SIZE, SIZE))
        x = random.randint(1, 10)
        if x == 1:
            image.save(f"{PATH_VALID_PNG}/{i:04d}.png")
            data_json(PATH_VALID_JSON, new_data)
        else:
            image.save(f"{PATH_TRAIN_PNG}/{i:04d}.png")
            data_json(PATH_TRAIN_JSON, new_data)
        bh.client.advance_frame()
        i += 1
print("\nFin de la capture")
