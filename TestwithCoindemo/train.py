from ultralytics import YOLO


# Define the path to the dataset.yaml file
dataset_yaml = "dataset.yaml"  # Path to your dataset.yaml file


model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)


# Train the model
results = model.train(data="dataset.yaml", epochs=10, imgsz=640)

