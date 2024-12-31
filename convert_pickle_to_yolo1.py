import os
import pickle
import shutil
from PIL import Image

# File paths
animal_class_path = './AwA-Pose/Annotations/Animal_Class.txt'
animal_data_path = './AwA-Pose/Annotations'
classes_txt_path = './AwA2-data/Animals_with_Attributes2/classes.txt'
image_root_path = './AwA2-data/Animals_with_Attributes2/JPEGImages'

yolo_labels_path = './yoloDataset/labels/'
yolo_images_path = './yoloDataset/images/'

# Function to read animal classes and their counts
def read_animal_classes():
    animal_classes = {}
    class_indices = {}  # To map class names to YOLO indices

    # Read the animal classes (name and count) from Animal_Class.txt
    with open(animal_class_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if idx >= 35:  # Stop after reading 35 lines
                break
            parts = line.strip().split('=')
            if len(parts) == 2:  # Ensure valid class data format
                class_name = (parts[0].split('.')[1].strip()).lower()  # Extract class name (after number)
                count = int(parts[1].strip())  # Get count as integer
                animal_classes[class_name] = count

    # Read the full list of all possible class names from 'classes.txt'
    with open(classes_txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split('\t')
            class_index = int(parts[0]) - 1  # Convert to 0-based index
            class_name = parts[1].strip()

            # Only include classes that exist in the animal_classes data
            # if class_name in animal_classes:
            class_indices[class_name] = class_index

    return animal_classes, class_indices

# Function to convert the pickle data to YOLO format
def convert_pickle_to_yolo(pickle_file, animal_class, image_path, class_indices):
    # Load the pickle data
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # Initialize the YOLO label list
    yolo_labels = []

    # Extract bounding box info
    bbox = data['a1']['bbox']
    if bbox[2] == -1 or bbox[3] == -1:
        return None  # Skip invalid annotations

    # Normalize bounding box
    img_width, img_height = Image.open(image_path).size
    x_center = (bbox[0] + bbox[2]) / 2.0 / img_width
    y_center = (bbox[1] + bbox[3]) / 2.0 / img_height
    width = (bbox[2] - bbox[0]) / img_width
    height = (bbox[3] - bbox[1]) / img_height

    # Add the class index and normalized bounding box
    yolo_labels.append(f"{class_indices[animal_class]} {x_center} {y_center} {width} {height}")
    
    return yolo_labels

# Function to process and convert the annotations
def process_animal_classes():
    # animal_classes, class_indices = read_animal_classes()

    # animal_classes = {'antelope': 283, 'bobcat': 304, 'buffalo': 242, 'chihuahua': 299, 'collie': 311, 'cow': 301, 'dalmatian': 304, 'deer': 301, 'elephant': 296, 'fox': 315, 'german+shepherd': 303, 'giant+panda': 302, 'giraffe': 293, 'grizzly+bear': 299, 'hippopotamus': 290, 'horse': 299, 'leopard': 302, 'lion': 290, 'moose': 303, 'otter': 294, 'ox': 302, 'persian+cat': 296, 'pig': 302, 'polar+bear': 302, 'rabbit': 315, 'raccon': 289, 'rhinoceros': 293, 'sheep': 281, 'siamese+cat': 261, 'skunk': 97, 'squirrel': 314, 'weasel': 202, 'wolf': 296, 'zebra': 287}
    animal_classes = {'wolf': 296, 'zebra': 287}
    class_indices = {'antelope': 0, 'grizzly+bear': 1, 'killer+whale': 2, 'beaver': 3, 'dalmatian': 4, 'persian+cat': 5, 'horse': 6, 'german+shepherd': 7, 'blue+whale': 8, 'siamese+cat': 9, 'skunk': 10, 'mole': 11, 'tiger': 12, 'hippopotamus': 13, 'leopard': 14, 'moose': 15, 'spider+monkey': 16, 'humpback+whale': 17, 'elephant': 18, 'gorilla': 19, 'ox': 20, 'fox': 21, 'sheep': 22, 'seal': 23, 'chimpanzee': 24, 'hamster': 25, 'squirrel': 26, 'rhinoceros': 27, 'rabbit': 28, 'bat': 29, 'giraffe': 30, 'wolf': 31, 'chihuahua': 32, 'rat': 33, 'weasel': 34, 'otter': 35, 'buffalo': 36, 'zebra': 37, 'giant+panda': 38, 'deer': 39, 'bobcat': 40, 'pig': 41, 'lion': 42, 'mouse': 43, 'polar+bear': 44, 'collie': 45, 'walrus': 46, 'raccoon': 47, 'cow': 48, 'dolphin': 49}
    
    for class_name, count in animal_classes.items():
        class_folder_path = os.path.join(animal_data_path, class_name)
        if not os.path.exists(class_folder_path):
            continue
        
        # Prepare to move labeled images and create label files
        for idx, pickle_file in enumerate(os.listdir(class_folder_path)):
            if not pickle_file.endswith('.pickle'):
                continue
            
            # Define paths
            pickle_file_path = os.path.join(class_folder_path, pickle_file)
            image_name = pickle_file.replace('.pickle', '.jpg')
            image_path = os.path.join(image_root_path, class_name, image_name)
            
            # Check if there is a corresponding image
            if not os.path.exists(image_path):
                continue  # Skip if no corresponding image
            
            # Convert pickle to YOLO format
            yolo_labels = convert_pickle_to_yolo(pickle_file_path, class_name, image_path, class_indices)
            
            if yolo_labels:
                # Save YOLO labels
                label_file_path = os.path.join(yolo_labels_path, pickle_file.replace('.pickle', '.txt'))
                with open(label_file_path, 'w') as f:
                    f.write("\n".join(yolo_labels))

                # Move the corresponding image to the yoloDataset images folder
                shutil.copy(image_path, os.path.join(yolo_images_path, image_name))

# Run the process
if __name__ == "__main__":
    process_animal_classes()
