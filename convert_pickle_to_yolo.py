import os
import pickle
import numpy as np

# Define paths
annotations_path = './AwA-Pose/Annotations/'
yolo_labels_path = './yoloDataset/labels/'
class_names_path = './AwA-Pose/Annotations/class_names.txt'
animal_class_path = './AwA-Pose/Annotations/Animal_Class.txt'

# Function to read class names
def read_class_names():
    with open(class_names_path, 'r') as f:
        # Skip the first line (_background_) and read the rest
        return [line.strip() for line in f.readlines()[1:]]

# Function to read animal classes and their counts
def read_animal_classes():
    animal_classes = {}
    with open(animal_class_path, 'r') as f:
        for line in f.readlines():
            # Parse class and count
            parts = line.strip().split('=')
            class_name = parts[0].strip()
            count = int(parts[1].strip())
            animal_classes[class_name] = count
    return animal_classes

# Function to convert bounding box from pickle file to YOLO format
def convert_bbox_to_yolo(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    # YOLO format: x_center, y_center, width, height (normalized by image dimensions)
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

# Function to convert keypoints from pickle file to YOLO format
def convert_keypoints_to_yolo(keypoints, img_width, img_height):
    # Extract points and normalize them
    keypoint_list = []
    for key in keypoints:
        if keypoints[key] != [-1, -1]:  # Ignore missing keypoints
            px, py = keypoints[key]
            px_normalized = px / img_width
            py_normalized = py / img_height
            keypoint_list.append(px_normalized)
            keypoint_list.append(py_normalized)
    return keypoint_list

# Main function to process files and generate YOLO labels
def process_animal_classes():
    class_names = read_class_names()
    animal_classes = read_animal_classes()

    for animal_class, count in animal_classes.items():
        animal_folder = os.path.join(annotations_path, animal_class.lower())
        label_folder = os.path.join(yolo_labels_path, animal_class.lower())
        os.makedirs(label_folder, exist_ok=True)
        
        # Read pickle files from the class folder
        pickle_files = [f for f in os.listdir(animal_folder) if f.endswith('.pkl')]
        
        for i, pickle_file in enumerate(pickle_files):
            if i >= count:
                break

            pickle_path = os.path.join(animal_folder, pickle_file)
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)

            # Extract bounding box and keypoints
            bbox = data['bbox']
            keypoints = {key: data[key] for key in data if key != 'bbox'}

            # Get image dimensions (assuming all images are the same size)
            img_width, img_height = 1000, 1000  # Replace with actual image dimensions

            # Convert bounding box to YOLO format
            x_center, y_center, width, height = convert_bbox_to_yolo(bbox, img_width, img_height)

            # Convert keypoints to YOLO format
            keypoint_list = convert_keypoints_to_yolo(keypoints, img_width, img_height)

            # Get the class index from Animal_Class.txt
            class_index = list(animal_classes.keys()).index(animal_class) + 1  # YOLO class index starts from 1

            # Save YOLO formatted label to a .txt file
            yolo_label_path = os.path.join(label_folder, pickle_file.replace('.pkl', '.txt'))
            with open(yolo_label_path, 'w') as label_file:
                label_file.write(f"{class_index} {x_center} {y_center} {width} {height} ")
                label_file.write(' '.join(map(str, keypoint_list)) + '\n')

if __name__ == '__main__':
    process_animal_classes()
