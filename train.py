from ultralytics import YOLO

# model = YOLO("yolov8n-pose.pt")

model = YOLO("runs/pose/train2/weights/last.pt")


results = model.train(data="config.yaml", epochs=470, imgsz=640)
