import cv2
import numpy as np
from collections import deque
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

# Buffer to store keypoint positions (second keypoint - club base)
trajectory = deque(maxlen=20)  # Store last 20 positions for smoothing

def moving_average(points, window_size=5):
    """Applies a simple moving average filter to smooth the trajectory."""
    if len(points) < window_size:
        return points[-1]  # Not enough points, return last known position
    avg_x = np.mean([p[0] for p in points[-window_size:]])
    avg_y = np.mean([p[1] for p in points[-window_size:]])
    return int(avg_x), int(avg_y)

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

    # Add detected keypoint to trajectory, or keep last known if missing
    if detected_kp:
        trajectory.append(detected_kp)
    elif trajectory:  # Use last known position if keypoint is lost
        trajectory.append(trajectory[-1])

    # Smooth the trajectory using moving average
    if trajectory:
        smoothed_kp = moving_average(list(trajectory))

        # Draw trajectory line
        for i in range(1, len(trajectory)):
            pt1 = moving_average(list(trajectory)[:i])  # Smooth previous points
            pt2 = moving_average(list(trajectory)[:i+1])
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

        # Draw smoothed point
        cv2.circle(frame, smoothed_kp, 6, (255, 255, 0), -1)

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
