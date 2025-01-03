from ultralytics import YOLO

# model = YOLO("yolov8n-pose.pt")

model = YOLO("runs/pose/train7/weights/best.pt")


results = model.train(data="config.yaml", epochs=25, imgsz=640)


# train:    5
# train1:   25
# train3:   470
# train4:   25
# train5:   75
# train6:   10   (used best.pt this time)
# 确实该用best.pt
# train7:   40  best

# start to evaluate the model, whether to continue training
# train8:   25  best

# ---done---