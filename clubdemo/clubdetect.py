# use torch to decode yolo result: GPU
# numpy array: CPU

import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model_path = "7best.pt"
model = YOLO(model_path)

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Perform inference
    results = model(frame)

    print(results)
    breakpoint()
    
    # Extract keypoints and bounding boxes
    for result in results:
        boxes = result.boxes.cpu().numpy()
        keypoints = result.keypoints.cpu().numpy()
        
        for box, kp in zip(boxes, keypoints):
            x1, y1, x2, y2 = map(int, box[:4])
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw keypoints
            for (kp_x, kp_y) in kp:
                cv2.circle(frame, (int(kp_x), int(kp_y)), 3, (255, 0, 0), -1)
    
    # Display the frame with detections
    cv2.imshow('Club Detections', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
