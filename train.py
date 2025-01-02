from ultralytics import YOLO

# model = YOLO("yolov8n-pose.pt")

model = YOLO("runs/pose/train4/weights/last.pt")


results = model.train(data="config.yaml", epochs=75, imgsz=640)


# train:    5
# train1:   25
# train3:   470
# train4:   25
# train5:   75