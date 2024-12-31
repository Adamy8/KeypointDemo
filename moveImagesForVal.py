import os
import random
import shutil

# Paths
yolo_dataset_path = "yoloDataset"
images_path = os.path.join(yolo_dataset_path, "images")
labels_path = os.path.join(yolo_dataset_path, "labels")
val_images_path = os.path.join(yolo_dataset_path, "val", "images")
val_labels_path = os.path.join(yolo_dataset_path, "val", "labels")

# Create directories for validation set
os.makedirs(val_images_path, exist_ok=True)
os.makedirs(val_labels_path, exist_ok=True)

# Percentage of files to move to validation set
val_percentage = 0.2

# Get all image files and corresponding label files
image_files = sorted([f for f in os.listdir(images_path) if f.endswith(".jpg")])
label_files = sorted([f for f in os.listdir(labels_path) if f.endswith(".txt")])

# Ensure that label files match the image files
image_files_base = {os.path.splitext(f)[0] for f in image_files}
label_files_base = {os.path.splitext(f)[0] for f in label_files}
valid_basenames = image_files_base.intersection(label_files_base)

# Filter the files to only include those with matching basenames
image_files = [f"{name}.jpg" for name in valid_basenames]
label_files = [f"{name}.txt" for name in valid_basenames]

# Randomly select a subset of files for validation
val_count = int(len(image_files) * val_percentage)
val_indices = random.sample(range(len(image_files)), val_count)

# Move selected files to validation set
for idx in val_indices:
    image_file = image_files[idx]
    label_file = label_files[idx]
    
    # Move image
    shutil.move(os.path.join(images_path, image_file), os.path.join(val_images_path, image_file))
    
    # Move label
    shutil.move(os.path.join(labels_path, label_file), os.path.join(val_labels_path, label_file))

print(f"Moved {val_count} files to validation set.")
