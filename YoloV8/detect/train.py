from ultralytics import YOLO

if __name__ == '__main__':
    
    model = YOLO('yolov8n.yaml')

    results = model.train(data='YoloV8\detect\data.yaml', epochs=100, imgsz=640, device=0)