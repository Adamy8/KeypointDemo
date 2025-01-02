import os
import shutil
import random

# Paths to the train, val, and test directories
train_img_dir = './yoloDataset/train/images/'
train_label_dir = './yoloDataset/train/labels/'
val_img_dir = './yoloDataset/val/images/'
val_label_dir = './yoloDataset/val/labels/'
test_img_dir = './yoloDataset/test/images/'
test_label_dir = './yoloDataset/test/labels/'

# Create the val and test directories if they don't exist
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# Get a list of all the image files in the train directory
image_files = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]

# Randomly shuffle the list of image files
random.shuffle(image_files)

# Calculate how many files to move for 15% for val and 15% for test
val_count = int(0.1 * len(image_files))
test_count = int(0.1 * len(image_files))

# Split the image files into val and test sets
val_files = image_files[:val_count]
test_files = image_files[val_count:val_count + test_count]
train_files = image_files[val_count + test_count:]

# Move files to val and test sets and corresponding label files
def move_files(file_list, img_src_dir, label_src_dir, img_dst_dir, label_dst_dir):
    for file_name in file_list:
        # Move image file
        img_src = os.path.join(img_src_dir, file_name)
        img_dst = os.path.join(img_dst_dir, file_name)
        shutil.move(img_src, img_dst)

        # Move corresponding label file
        label_file_name = file_name.replace('.jpg', '.txt')
        label_src = os.path.join(label_src_dir, label_file_name)
        label_dst = os.path.join(label_dst_dir, label_file_name)
        shutil.move(label_src, label_dst)

# Move 15% of the files to val and test
move_files(val_files, train_img_dir, train_label_dir, val_img_dir, val_label_dir)
move_files(test_files, train_img_dir, train_label_dir, test_img_dir, test_label_dir)

print(f"Moved {val_count} images to validation and {test_count} images to test.")
