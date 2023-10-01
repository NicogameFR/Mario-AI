
from ultralytics import YOLO
import pyscreenshot
# Load a pretrained YOLOv8n model


model = YOLO('best.pt')

# Define current screenshot as source

# Run inference on the source
while True:
    image = pyscreenshot.grab(childprocess=False, bbox = (80, 46, 863, 659))
    results = model(image, show = True)  # list of Results objects