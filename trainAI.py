from ultralytics import YOLO
import pyscreenshot
import numpy as np

model_detect = YOLO('YoloV8/detect/best.pt')
model_class = YOLO('YoloV8/class/best.pt')
matrice_zeros = np.empty((256, 224), dtype=object)

running = True


def get_box(results):
    boxe = results[0].boxes.xyxyn.cpu().numpy()
    cls = results[0].boxes.cls.cpu().numpy()
    for i in range(len(boxe)):
        x1 = round(boxe[i][0]*256)
        y1 = round(boxe[i][1]*224)
        x2 = round(boxe[i][2]*256)
        y2 = round(boxe[i][3]*224)
        x_ecart = x2 - x1
        y_ecart = y2 - y1
        for j in range(x_ecart):
            for k in range(y_ecart):
                matrice_zeros[x1+j][y1+k] = int(cls[i])
    return matrice_zeros


# 2 = vie
# 0 = fin
# 1 = mort
while running:
    # image = pyscreenshot.grab(childprocess=False, bbox = (1167, 66, 1712, 483))
    image = pyscreenshot.grab(childprocess=False, bbox=(1515, 54, 2324, 673))
    img = image.resize((640, 640))
    results = model_class(img, show=True, device=0)
    if results[0].probs.top1 == 2:
        results = model_detect(img, show=False, device=0)
        result = get_box(results)
    else:
        pass
    running = True
