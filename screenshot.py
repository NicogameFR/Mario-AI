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
#%%
nom_fichier = "Mario_World.csv"
path = "./worlds"
if not os.path.exists(path):
    os.mkdir(path)
i = 0


bh.memory.add_address(0x04AC) # Mario's x1 position
bh.memory.add_address(0x04AD) # Mario's y1 position
bh.memory.add_address(0x04AE) # Mario's x2 position
bh.memory.add_address(0x04AF) # Mario's y2 position
while bh.client.is_connected():
    
    image = pyscreenshot.grab(childprocess=False, bbox=(1167, 66, 1712, 483))
    image = image.resize((64, 64))
    image = image.convert("L")  # '1' pour carré noir et blanc
    image.save(f"./{path}/world_{i}.png")
    bh.memory.request_memory()
    print(str(bh.memory.get_memory())+"\r", end="")
    x1 = bh.memory.get_memory()["0x4ac"]
    y1 = bh.memory.get_memory()["0x4ad"]
    x2 = bh.memory.get_memory()["0x4ae"]
    y2 = bh.memory.get_memory()["0x4af"]
    datas = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    bh.client.advance_frame()
    i+= 1