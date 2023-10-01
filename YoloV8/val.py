from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Define source as YouTube video URL
source = 'https://youtu.be/LNwODJXcvt4'

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects