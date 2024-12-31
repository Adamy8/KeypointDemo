import os
import pickle
from PIL import Image
import shutil

# Define paths
annotations_path = './AwA-Pose/Annotations/'
yolo_labels_path = './yoloDataset/labels/'
class_names_path = './AwA-Pose/Annotations/class_names.txt'
animal_class_path = './AwA-Pose/Annotations/Animal_Class.txt'
images_path = './AwA2-data/Animals_with_Attributes2/JPEGImages/'
yolo_images_path = './yoloDataset/images/'

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
            # Remove the number and period prefix (e.g., '1.' -> '')
            parts = line.strip().split('=')  # Split on the '=' to separate class name and count
            class_name = parts[0].split('.')[1].strip()  # Remove the prefix like '1.' and get the class name
            count = int(parts[1].strip())  # Parse the count
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

    # animal_classes = read_animal_classes()
    animal_classes = {'antelope': 1}  # For testing with the first 'antelope' only

    for animal_class, count in animal_classes.items():
        animal_folder = os.path.join(annotations_path, animal_class.lower())
        label_folder = os.path.join(yolo_labels_path, animal_class.lower())
        image_folder = os.path.join(images_path, animal_class.lower())  # Path to the class's images
        os.makedirs(label_folder, exist_ok=True)
        os.makedirs(os.path.join(yolo_images_path, animal_class.lower()), exist_ok=True)  # Directory for the moved images
        
        # Read pickle files from the class folder
        pickle_files = [f for f in os.listdir(animal_folder) if f.endswith('.pickle')]
        
        for i, pickle_file in enumerate(pickle_files):
            if i >= count:
                break

            pickle_path = os.path.join(animal_folder, pickle_file)
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            # Access the 'a1' key
            animal_data = data['a1']

            # Extract bounding box and keypoints
            bbox = animal_data['bbox']
            keypoints = {key: animal_data[key] for key in animal_data if key != 'bbox'}

            # Get corresponding image
            image_name = pickle_file.replace('.pickle', '.jpg')
            image_path = os.path.join(image_folder, image_name)

            # if os.path.exists(image_path):
            # Open the image and get dimensions
            with Image.open(image_path) as img:
                img_width, img_height = img.size  # Get actual image width and height

            # Convert bounding box to YOLO format
            x_center, y_center, width, height = convert_bbox_to_yolo(bbox, img_width, img_height)

            # Convert keypoints to YOLO format
            keypoint_list = convert_keypoints_to_yolo(keypoints, img_width, img_height)

            # Get the class index (always 1 for 'antelope' in this test)
            class_index = animal_classes.get(animal_class.lower(), -1)

            # Save YOLO formatted label to a .txt file
            yolo_label_path = os.path.join(label_folder, pickle_file.replace('.pickle', '.txt'))
            with open(yolo_label_path, 'w') as label_file:
                label_file.write(f"{class_index} {x_center} {y_center} {width} {height} ")
                label_file.write(' '.join(map(str, keypoint_list)) + '\n')

            # Move the corresponding image to the yoloDataset/images/ directory
            yolo_image_path = os.path.join(yolo_images_path, animal_class.lower(), image_name)
            shutil.move(image_path, yolo_image_path)  # Move image to yoloDataset/images/


if __name__ == '__main__':
    process_animal_classes()
