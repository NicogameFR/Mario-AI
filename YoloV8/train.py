from ultralytics import YOLO

def train():
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data='data.yaml', epochs=100, imgsz=640, device=0)
    return results

if __name__ == '__main__':
    result = train()