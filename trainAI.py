from ultralytics import YOLO
import pyscreenshot
import numpy as np
import birdseyelib as bird
import keyboard
import random
import time
import os


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


class Object:

    def __init__(self, x1, y1, x2, y2, classe, objects):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.classe = int(classe)
        self.objects = objects

    def __str__(self):
        return f"Object: {CLASSES[self.classe]}"

    def __repr__(self):
        return f"Object: {CLASSES[self.classe]}"


CLASSES = [
    "big_mario",
    "brick_block",
    "coin",
    "empty_block",
    "fire_mario",
    "fireball",
    "flower",
    "goal_pole",
    "goomba",
    "hard_block",
    "koopa",
    "little_mario",
    "mushroom",
    "mystery_block",
    "pipe",
    "pipe_head",
    "shell",
    "undestructible_block",
]
model_detect = YOLO("YoloV8/detect/best.pt")

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
        x1 = round(boxe[i][0] * 256)
        y1 = round(boxe[i][1] * 224)
        x2 = round(boxe[i][2] * 256)
        y2 = round(boxe[i][3] * 224)

        object_list.append(Object(x1, y1, x2, y2, cls[i], object_list))
    return object_list


def input_():
    x = random.randint(0, 1)
    if x == 0:
        return False
    else:
        return True


actions = []
start = True
time_in_life = 0
while bh.client.is_connected():
    # while start:
    a = input_()
    b = input_()
    up = input_()
    down = input_()
    right = input_()
    left = input_()
    actions.append([a, b, up, down, right, left])
    bh.memory.request_memory()
    for i in range(10):
        bh.controller_input.set_controller_input(a, b, up, down, right, left)
        bh.client.advance_frame()
    if (
        bh.memory.get_memory()["0xe"] == 11
        or bh.memory.get_memory()["0xe"] == 6
        or bh.memory.get_memory()["0xe"] == 0
    ):
        keyboard.press_and_release("f4")
    # image = pyscreenshot.grab(childprocess=False, bbox=(1167, 66, 1712, 483))
    image = pyscreenshot.grab(childprocess=False, bbox=(1515, 54, 2324, 673))
    img = image.resize((640, 640))
    results = model_detect(img, show=False, device=0)
    results = get_box(results)
    # print(results)
    for result in results:
        if result.classe in [0, 4, 11]:
            time_in_life += 1
    print(time_in_life + "\r", end="")
    start = True
