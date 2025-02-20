# this file cannot work, yolo result format not matched

import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model_path = "7best.pt"
model = YOLO(model_path)

# Load the image
image_path = "./images.jpeg"
image = cv2.imread(image_path)

# Perform inference
results = model(image)

print(results[0].keypoints)
breakpoint()


# Extract keypoints and bounding boxes
for result in results:
    boxes = result.boxes
    keypoints = result.keypoints
    
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        for box in xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            conf = box[4]
            cls = int(box[5])
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Display class and confidence
            label = f'Class: {cls}, Conf: {conf:.2f}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if keypoints is not None and keypoints.has_visible:
        kps = keypoints.xy.cpu().numpy()
        for kp in kps[0]:
            x, y = map(int, kp[:2])
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

# Display the image with detections
cv2.imshow('YOLO-Pose Detections', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
