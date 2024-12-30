from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)


# Train the model
results = model.train(data="dataset.yaml", epochs=1, imgsz=640)

