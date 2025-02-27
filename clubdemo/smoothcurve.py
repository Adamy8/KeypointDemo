import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Load the YOLO model
model_path = "7best.pt"
model = YOLO(model_path)

# Open input video
input_video_path = "input.mp4"
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
output_video_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Buffer to store keypoint positions (second keypoint - club base)
trajectory = deque(maxlen=75)  # Increased history length for smoother motion

# Initialize EMA smoothing parameters
ema_position = None  # Exponential Moving Average position
alpha = 0.05  # Smoothing factor (higher = more reactive, lower = smoother)

def exponential_moving_average(new_point):
    """Applies an exponential moving average filter for smoothing."""
    global ema_position
    if ema_position is None:
        ema_position = new_point  # Initialize with first detected position
    else:
        ema_position = (
            int(alpha * new_point[0] + (1 - alpha) * ema_position[0]),
            int(alpha * new_point[1] + (1 - alpha) * ema_position[1])
        )
    return ema_position

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Perform inference
    results = model(frame)
    
    detected_kp = None  # Store detected keypoint
    
    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints

        if boxes is not None and len(boxes) > 0:
            best_box = boxes[0]  # YOLO detections are ranked by confidence
            x1, y1, x2, y2 = map(int, best_box.xyxy.cpu().numpy()[0])
            conf = best_box.conf.cpu().numpy()[0]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'Conf: {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw keypoints and track the second keypoint (club base)
            if keypoints is not None and keypoints.has_visible:
                kps = keypoints.xy.cpu().numpy()[0]  # Take first keypoint set
                if len(kps) > 1:  # Ensure the second keypoint exists
                    x, y = map(int, kps[1][:2])  # Extract second keypoint (club base)
                    detected_kp = (x, y)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # Add detected keypoint to trajectory, or estimate its position if missing
    if detected_kp:
        smoothed_kp = exponential_moving_average(detected_kp)
        trajectory.append(smoothed_kp)
    elif trajectory:  # If keypoint is lost, use last known EMA position
        trajectory.append(trajectory[-1])

    # Draw smooth trajectory line
    if len(trajectory) > 1:
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)

    # Draw latest smoothed keypoint
    if trajectory:
        cv2.circle(frame, trajectory[-1], 6, (255, 255, 0), -1)

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
