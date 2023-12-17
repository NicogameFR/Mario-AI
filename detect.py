from ultralytics import YOLO
import pyscreenshot
import pygame




GRID_WIDTH = 255
GRID_HEIGHT = 223
CELL_SIZE = 3


model = YOLO('best.pt')

pygame.init()
fenetre = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))

pygame.display.set_caption("Ce que voit l'IA")

running = True



def get_box(results):
    boxe = results[0].boxes.xywhn.cpu().numpy()
    cls = results[0].boxes.cls.cpu().numpy()
    for i in range(len(boxe)):
        if cls[i] == 17  or cls[i] == 9:
            color = (0, 0, 0)
        elif cls[i] == 13:
            color = (255, 255, 0)
        elif cls[i] == 11:
            color = (255, 0, 0)
        elif cls[i] == 14  or cls[i] == 1:
            color = (0, 255, 0)
        elif cls[i] == 8 or cls[i] == 1:
            color = (153, 79, 0)
        else:
            color = (0 , 255, 255)
        pygame.draw.rect(fenetre, color, (boxe[i][0]*GRID_WIDTH * CELL_SIZE, boxe[i][1]*GRID_HEIGHT * CELL_SIZE, boxe[i][2]*GRID_WIDTH * CELL_SIZE, boxe[i][3]*GRID_HEIGHT * CELL_SIZE))


while running:
    #image = pyscreenshot.grab(childprocess=False, bbox = (1167, 66, 1712, 483))
    image = pyscreenshot.grab(childprocess=False, bbox = (1515, 54, 2324, 673))
    img = image.resize((640, 640))
    results = model(img, show = False, device=0)
    fenetre.fill((147, 144, 254))
    get_box(results)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    pygame.display.flip()

pygame.quit()