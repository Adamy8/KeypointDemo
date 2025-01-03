from ultralytics import YOLO
import cv2

model_path = "runs/pose/train7/weights/last.pt"
model = YOLO(model_path)

keypoint_connections = []

video_path = "kubo_video.mp4"


# Function to draw the lines between keypoints
def draw_keypoints_and_lines(image, keypoints, connections):
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5:  # Only draw keypoints with confidence above 0.5
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green dots for keypoints

    # Draw lines between connected keypoints
    for start, end in connections:
        if keypoints[start][2] > 0.5 and keypoints[end][2] > 0.5:  # Confidence threshold
            start_point = (int(keypoints[start][0]), int(keypoints[start][1]))
            end_point = (int(keypoints[end][0]), int(keypoints[end][1]))
            cv2.line(image, start_point, end_point, (0, 0, 255), 2)  # Red lines for connections

    return image

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video details (like width, height, and FPS)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter to save the output video
output_path = "testResult.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Perform inference using YOLO model (keypoints are part of the output)
    results = model(frame)  # Perform inference

    # Get the keypoints from the results (adjust based on your model output)
    for result in results:
        keypoints = result.keypoints.xy  # Example: Extract the keypoints
        frame_with_lines = draw_keypoints_and_lines(frame.copy(), keypoints, keypoint_connections)

    # Write the frame to the output video
    out.write(frame_with_lines)

    # Display the frame (optional)
    cv2.imshow("Animal Pose Estimation", frame_with_lines)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()