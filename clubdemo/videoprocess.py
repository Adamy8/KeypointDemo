import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model_path = "7best.pt"
model = YOLO(model_path)

# Open input video
input_video_path = "input.mp4"  # Change this to your input video file
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
output_video_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Perform inference
    results = model(frame)
    
    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints

        if boxes is not None and len(boxes) > 0:
            # Get the most confident detection (first one)
            best_box = boxes[0]  # YOLO detections are ranked by confidence
            x1, y1, x2, y2 = map(int, best_box.xyxy.cpu().numpy()[0])
            conf = best_box.conf.cpu().numpy()[0]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'Conf: {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw keypoints for the most confident detection
            if keypoints is not None and keypoints.has_visible:
                kps = keypoints.xy.cpu().numpy()[0]  # Take first keypoint set
                for kp in kps:
                    x, y = map(int, kp[:2])
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    
    # Show frame while processing
    cv2.imshow("Processing Video", frame)

    # Write the processed frame to the output video
    out.write(frame)

    # Press 'q' to stop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video saved as {output_video_path}")
