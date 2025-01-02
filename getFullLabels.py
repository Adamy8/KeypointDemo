import os
import pickle
from PIL import Image


yolo_images_path = './yoloDataset/images/train/'
yolo_labels_path = './yoloDataset/labels/train/'

annotation_path = './AwA-Pose/Annotations'

animal_classes = ['antelope', 'bobcat', 'buffalo', 'chihuahua', 'collie', 'cow', 'dalmatian', 'deer', 'elephant', 'fox', 'german+shepherd', 'giant+panda', 'giraffe', 'grizzly+bear', 'hippopotamus', 'horse', 'leopard', 'lion', 'moose', 'otter', 'ox', 'persian+cat', 'pig', 'polar+bear', 'rabbit', 'raccon', 'rhinoceros', 'sheep', 'siamese+cat', 'skunk', 'squirrel', 'weasel', 'wolf', 'zebra']
class_indices = {'antelope': 0, 'grizzly+bear': 1, 'killer+whale': 2, 'beaver': 3, 'dalmatian': 4, 'persian+cat': 5, 'horse': 6, 'german+shepherd': 7, 'blue+whale': 8, 'siamese+cat': 9, 'skunk': 10, 'mole': 11, 'tiger': 12, 'hippopotamus': 13, 'leopard': 14, 'moose': 15, 'spider+monkey': 16, 'humpback+whale': 17, 'elephant': 18, 'gorilla': 19, 'ox': 20, 'fox': 21, 'sheep': 22, 'seal': 23, 'chimpanzee': 24, 'hamster': 25, 'squirrel': 26, 'rhinoceros': 27, 'rabbit': 28, 'bat': 29, 'giraffe': 30, 'wolf': 31, 'chihuahua': 32, 'rat': 33, 'weasel': 34, 'otter': 35, 'buffalo': 36, 'zebra': 37, 'giant+panda': 38, 'deer': 39, 'bobcat': 40, 'pig': 41, 'lion': 42, 'mouse': 43, 'polar+bear': 44, 'collie': 45, 'walrus': 46, 'raccoon': 47, 'cow': 48, 'dolphin': 49}
parts = [
    'nose', 'upper_jaw', 'lower_jaw', 'mouth_end_right', 'mouth_end_left',
    'right_eye', 'right_earbase', 'right_earend', 'right_antler_base', 'right_antler_end',
    'left_eye', 'left_earbase', 'left_earend', 'left_antler_base', 'left_antler_end',
    'neck_base', 'neck_end', 'throat_base', 'throat_end', 'back_base', 'back_end',
    'back_middle', 'tail_base', 'tail_end', 'front_left_thai', 'front_left_knee',
    'front_left_paw', 'front_right_thai', 'front_right_paw', 'front_right_knee',
    'back_left_knee', 'back_left_paw', 'back_left_thai', 'back_right_thai', 'back_right_paw',
    'back_right_knee', 'belly_bottom', 'body_middle_right', 'body_middle_left'
]


def generateCorrectLabel():
    # for animal in animal_classes:
    #     pickle_file_path = os.path.join(annotation_path,animal)
    for image_name in os.listdir(yolo_images_path):
        if not image_name.endswith(".jpg"):
            continue
        
        # image_name is in form: antelope_10002.jpg
        animal_name = image_name.split('_')[0]

        pickle_name = image_name.replace('.jpg','.pickle')
        pickle_file_path = os.path.join(annotation_path,animal_name,pickle_name)

        #theordically, every path have to exist, so i comment the exist test
        # if not os.path.exists(pickle_file_path):
        #     continue

        image_path = os.path.join(yolo_images_path,image_name)
        yolo_label = convert_pickle_to_yolo(pickle_file_path,animal_name,image_path)

        if yolo_label:
            # Save YOLO labels
            label_file_path = os.path.join(yolo_labels_path, pickle_name.replace('.pickle', '.txt'))
            with open(label_file_path, 'w') as f:
                f.write(" ".join(yolo_label))



def convert_pickle_to_yolo(pickle_file, animal_class, image_path):
    # Load the pickle data
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # Initialize the YOLO label list
    yolo_labels = []

    # Extract bounding box info
    bbox = data['a1']['bbox']
    # if bbox[2] == -1 or bbox[3] == -1:
    #     return None  # Skip invalid annotations

    # Normalize bounding box
    img_width, img_height = Image.open(image_path).size
    x_center = (bbox[0] + bbox[2]) / 2.0 / img_width
    y_center = (bbox[1] + bbox[3]) / 2.0 / img_height
    width = (bbox[2] - bbox[0]) / img_width
    height = (bbox[3] - bbox[1]) / img_height

    # Add the class index and normalized bounding box
    yolo_labels.append(f"{class_indices[animal_class]} {x_center} {y_center} {width} {height}")
    
    for part_name in parts:
        part_data = data['a1'][part_name]
        # yolo_labels.append(f"{part_data[0]:.6f} {part_data[1]:.6f}")  # make it 6 decimal places
        if part_data[0] != -1 and part_data[1] != -1:
            if is_keypoint_in_box(part_data, bbox):
                yolo_labels.append(f"{part_data[0] / img_width:.6f} {part_data[1] / img_height:.6f} 2")   # normalized!!!
            else:
                yolo_labels.append(f"{part_data[0] / img_width:.6f} {part_data[1] / img_height:.6f} 1")   # invisible
        else:
            yolo_labels.append("0 0 0")

    return yolo_labels



def is_keypoint_in_box(keypoint, bbox):
    keypoint_x, keypoint_y = keypoint
    x_min, y_min, x_max, y_max = bbox
    return x_min <= keypoint_x <= x_max and y_min <= keypoint_y <= y_max


if __name__ == "__main__":
    generateCorrectLabel()