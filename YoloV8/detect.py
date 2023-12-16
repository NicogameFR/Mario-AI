from ultralytics import YOLO
import pyscreenshot
# Load a pretrained YOLOv8n model


model = YOLO('C:/Users/bapti/Desktop/Mario-AI/YoloV8/best.pt')

# Define current screenshot as source

# Run inference on the source
while True:
    image = pyscreenshot.grab(childprocess=False, bbox = (1515, 54, 2324, 673))
    img = image.resize((640, 640))
    results = model(image, show = True, device=0)  # list of Results objects