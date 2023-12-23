from ultralytics import YOLO

if __name__ == '__main__':
    
    model = YOLO('yolov8n-cls.yaml')

    results = model.train(data='class', epochs=50, imgsz=640, device=0)