# %%
import pyscreenshot
import birdseyelib as bird
import os

# %%

HOST = "127.0.0.1"
PORT = 8080

PATH_PNG = "../.data/Mario_data/"

TOTAL_FRAMES = 5000
SIZE = 640

# %%


class BizHawk:

    def __init__(self, HOST, PORT):
        self.client = bird.Client(HOST, PORT)
        self.memory = bird.Memory(self.client)
        self.controller_input = bird.ControllerInput(self.client)
        self.emulation = bird.Emulation(self.client)
        self.external_tool = bird.ExternalTool(self.client)
        self.client.connect()
        self.external_tool.set_commandeer(True)
    def __str__(self):
        return f"BizHawk Client: {self.client}"


bh = BizHawk(HOST, PORT)
print("Connecting to server at {} on port {}.".format(HOST, PORT))


# %%
try:
    if os.listdir(PATH_PNG):
        os.system(f"cd {PATH_PNG} & del /f *.png")
        print("Suppression des anciennes captures")
except FileNotFoundError:
    PATH_PNG = ".data/Mario_data/"
    if os.listdir(PATH_PNG):
        os.system(f"cd {PATH_PNG} & del /f *.png")
        print("Suppression des anciennes captures")
# %%

i = 1

bh.emulation.request_framecount()
bh.client.advance_frame()
while bh.client.is_connected() and i <= TOTAL_FRAMES:
    print(
        f"Nombre de captures: {i}/{TOTAL_FRAMES}\r",
        end="",
    )
    image = pyscreenshot.grab(childprocess=False, bbox=(1167, 66, 1712, 483))
    #image = pyscreenshot.grab(childprocess=False, bbox=(1515, 54, 1809, 619))
    image = image.resize((SIZE, SIZE))
    image.save(f"{PATH_PNG}/{i:04d}.png")
    i += 1
    bh.emulation.request_framecount()
    bh.client.advance_frame()
print("\nFin de la capture")
