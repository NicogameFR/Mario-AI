from ultralytics import YOLO
import pyscreenshot
import numpy as np
import birdseyelib as bird
import keyboard
import random
import time


class BizHawk:

    def __init__(self, HOST, PORT):
        self.client = bird.Client(HOST, PORT)
        self.memory = bird.Memory(self.client)
        self.controller_input = bird.ControllerInput(self.client)
        self.emulation = bird.Emulation(self.client)
        self.external_tool = bird.ExternalTool(self.client)
        self.client.connect()

    def __str__(self) -> str:
        return f'BizHawk Client: {self.client}'


class Object:

    def __init__(self, x1, y1, x2, y2, classe, objects):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.classe = int(classe)
        self.objects = objects

    def __str__(self):
        return '\n' + str([CLASSES[self.classe], self.classe, self.x1, self.y1, self.x2, self.y2])

    def __repr__(self):
        return '\n' + str([CLASSES[self.classe], self.classe, self.x1, self.y1, self.x2, self.y2])


CLASSES = ['big_mario', 'brick_block', 'coin', 'empty_block', 'fire_mario', 'fireball', 'flower', 'goal_pole', 'goomba',
           'hard_block', 'koopa', 'little_mario', 'mushroom', 'mystery_block', 'pipe', 'pipe_head', 'shell', 'undestructible_block']

HOST = "127.0.0.1"
PORT = 8080


object_list = []


bh = BizHawk(HOST, PORT)
bh.memory.add_address(0x000E)
bh.external_tool.set_commandeer(True)

def get_box(results):
    boxe = results[0].boxes.xyxyn.cpu().numpy()
    cls = results[0].boxes.cls.cpu().numpy()
    for i in range(len(boxe)):
        x1 = round(boxe[i][0]*256)
        y1 = round(boxe[i][1]*224)
        x2 = round(boxe[i][2]*256)
        y2 = round(boxe[i][3]*224)
        object_list.append(Object(x1, y1, x2, y2, cls[i], object_list))
    return object_list


def input():
   x = random.randint(0, 1)
   if x == 0:
       return False
   else:
       return True




model_detect = YOLO('YoloV8/detect/best.pt')


while bh.client.is_connected():
    a = input()
    b = input()
    up = input()
    down = input()
    right = input()
    left = input()
    bh.memory.request_memory()
    for i in range(10):
        bh.controller_input.set_controller_input(a, b, up, down, right, left)
        bh.client.advance_frame()
    if bh.memory.get_memory()['0xe'] == 11 or bh.memory.get_memory()['0xe'] == 6 or bh.memory.get_memory()['0xe'] == 0:
       keyboard.press_and_release('f4')
    image = pyscreenshot.grab(childprocess=False, bbox=(1167, 66, 1712, 483))
    # image = pyscreenshot.grab(childprocess=False, bbox=(1515, 54, 2324, 673))
    img = image.resize((640, 640))
    results = model_detect(img, show=True, device=0)
    results = get_box(results)
