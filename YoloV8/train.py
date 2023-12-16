from ultralytics import YOLO

def train():
    # Load a model
    model = YOLO('YoloV8/MarioAI.yaml')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data='YoloV8/data.yaml', epochs=20, imgsz=640, device=0)
    return results

if __name__ == '__main__':
    result = train()