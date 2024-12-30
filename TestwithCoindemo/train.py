from ultralytics import YOLO


# Define the path to the dataset.yaml file
dataset_yaml = "dataset.yaml"  # Path to your dataset.yaml file


# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("runs/detect/train9/weights/last.pt")


# Train the model
# results = model.train(data="dataset.yaml", epochs=10, imgsz=640)
results = model.train(
    data="dataset.yaml", 
    epochs=200,
    imgsz=640, 
    # MAC
    # save_dir="/Users/AdamYE_1/Desktop/KeypointDemo/TestwithCoindemo/runs/detect/train",
    # project="/Users/AdamYE_1/Desktop/KeypointDemo/TestwithCoindemo/runs/detect"  # Explicitly set project directory

    # Windows
    save_dir="/home/ady/repos/KeypointDemo/TestwithCoindemo/runs/detect/train",
    project="/home/ady/repos/KeypointDemo/TestwithCoindemo/runs/detect"  # Explicitly set project directory
)

