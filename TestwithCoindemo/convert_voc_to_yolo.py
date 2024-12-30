#converting coinDemo dataset (.xml) to YOLO formatt

import os
import shutil
import xml.etree.ElementTree as ET

# Define class names (update with your actual classes)
class_names = ["quarter", "dime", "nickel", "penny"]  # Replace with your classes

def convert_voc_to_yolo(xml_path, img_width=1280, img_height=720):
    """
    Convert a Pascal VOC XML annotation to YOLO format.
    The default resolution of the images is assumed to be 1280x720.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    yolo_labels = []
    
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_id = class_names.index(class_name)  # Get the class index
        
        # Get bounding box coordinates
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        
        # Convert to YOLO format: x_center, y_center, width, height (normalized)
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        # Append the YOLO label
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_labels

def copy_images(folder, output_folder):
    """
    Copy images from the original folder to a new output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(folder):
        if file_name.endswith(".jpg"):  # Process only JPG images
            # Copy each image to the output folder
            src_image = os.path.join(folder, file_name)
            dst_image = os.path.join(output_folder, file_name)
            shutil.copy(src_image, dst_image)
            print(f"Copied: {file_name}")

def process_dataset(folder, output_folder, labels_folder):
    """
    Convert all XML annotations in the folder to YOLO format, 
    and copy all images to a new folder for training.
    """
    for file_name in os.listdir(folder):
        if file_name.endswith(".jpg"):  # Process only JPG images
            # Corresponding XML file should have the same name
            xml_file = os.path.join(folder, file_name.replace(".jpg", ".xml"))
            
            # Check if the corresponding XML file exists
            if os.path.exists(xml_file):
                # Convert XML to YOLO format
                yolo_labels = convert_voc_to_yolo(xml_file)
                
                # Save to a .txt file in the labels folder
                txt_file = os.path.join(labels_folder, file_name.replace(".jpg", ".txt"))
                with open(txt_file, "w") as f:
                    for label in yolo_labels:
                        f.write(f"{label}\n")
                print(f"Converted: {file_name.replace('.jpg', '.txt')}")
            else:
                print(f"Warning: No XML file found for {file_name}")
    
def create_folders(output_folder, labels_folder):
    """
    Create output folders if they don't exist.
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

if __name__ == "__main__":
    folder = "../../coinDemo/Pics"  # Path to your images and XMLs folder (relative path)
    output_folder = "dataset/copied_images"  # Folder to copy images to
    labels_folder = "dataset/yolo_labels"  # Folder to save YOLO .txt label files
    
    # Step 1: Create output folders if they don't exist
    create_folders(output_folder, labels_folder)
    
    # Step 2: Copy images to the new folder
    copy_images(folder, output_folder)
    
    # Step 3: Convert XML annotations to YOLO format and save them
    process_dataset(folder, output_folder, labels_folder)
    
    print("Conversion and image copying complete!")
