# %%

import pyscreenshot
import birdseyelib as bird
import os
import json

# %%


class BizHawk:

    def __init__(self, HOST, PORT):
        self.client = bird.Client(HOST, PORT)
        self.memory = bird.Memory(self.client)
        self.controller_input = bird.ControllerInput(self.client)
        self.emulation = bird.Emulation(self.client)
        self.external_tool = bird.ExternalTool(self.client)
        self.client.connect()

    def __str__(self) -> str:
        return f"BizHawk Client: {self.client}"


HOST = "127.0.0.1"
PORT = 8080
bh = BizHawk(HOST, PORT)

# try:
#   os.system("cd ./worlds/ & del /f *.png")
# except:
#   pass
# try:
#   os.system("del Mario_World.json")
# except:
#   pass

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
PATH_JSON = "Mario_World.json"
PATH_PNG = "./worlds"
if not os.path.exists(PATH_PNG):
    os.mkdir(PATH_PNG)
i = 0

bh.memory.add_address(0x04AC)  # Mario's x1 position
bh.memory.add_address(0x04AD)  # Mario's y1 position
bh.memory.add_address(0x04AE)  # Mario's x2 position
bh.memory.add_address(0x04AF)  # Mario's y2 position
bh.memory.add_address(0x071A)  # Décalage niveau
bh.memory.add_address(0x071C)  # Décalage écran

while bh.client.is_connected():
    bh.memory.request_memory()
    x1 = bh.memory.get_memory()["0x4ac"]
    y1 = bh.memory.get_memory()["0x4ad"]
    x2 = bh.memory.get_memory()["0x4ae"]
    y2 = bh.memory.get_memory()["0x4af"]
    if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1: 
        d1 = bh.memory.get_memory()["0x71a"]
        d2 = bh.memory.get_memory()["0x71c"]
        decalage = d1 * 256 + d2
        print(f"Decalage: {decalage}")
        new_data = {"Decalage": decalage, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        image = pyscreenshot.grab(childprocess=False, bbox=(1167, 66, 1712, 483))
        image = image.resize((64, 64))
        image = image.convert("L")  # '1' pour carré noir et blanc
        image.save(f"./{PATH_PNG}/world_{i}.png")
        data_json(PATH_JSON, new_data)
        bh.client.advance_frame()
        i += 1
