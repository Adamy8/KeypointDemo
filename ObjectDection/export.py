from ultralytics import YOLO

model = YOLO("yolo11nMY.pt")

model.export(format="onnx")