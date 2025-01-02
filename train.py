from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

# for later use
# model = YOLO("runs/detect/train/weights/last.pt")


results = model.train(data="config.yaml", epochs=2, imgsz=640)
