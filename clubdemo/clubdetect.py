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
    
    # Extract keypoints and bounding boxes
    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints
        
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            for box in xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                conf = boxes.conf.cpu().numpy()[0]
                # cls = int(box[5])    # class
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Display class and confidence
                label = f'Conf: {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if keypoints is not None and keypoints.has_visible:
            kps = keypoints.xy.cpu().numpy()
            for kp in kps[0]:
                x, y = map(int, kp[:2])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    
    # Display the frame with detections
    cv2.imshow('YOLO-Pose Detections', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
