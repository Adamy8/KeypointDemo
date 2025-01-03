from ultralytics import YOLO
import cv2

model_path = "runs/pose/train7/weights/last.pt"
model = YOLO(model_path)

keypoint_connections = [
    (0, 1),  # nose -> upper_jaw
    (1, 2),  # upper_jaw -> lower_jaw
    (3, 4),  # mouth_end_right -> mouth_end_left
    (5, 3),  # right_eye -> mouth_end_right
    (5, 4),  # right_eye -> mouth_end_left
    (6, 5),  # right_earbase -> right_eye
    (7, 6),  # right_earend -> right_earbase
    (8, 7),  # right_antler_base -> right_earend
    (9, 8),  # right_antler_end -> right_antler_base
    (10, 3), # left_eye -> mouth_end_right
    (10, 4), # left_eye -> mouth_end_left
    (11, 10), # left_earbase -> left_eye
    (12, 11), # left_earend -> left_earbase
    (13, 12), # left_antler_base -> left_earend
    (14, 13), # left_antler_end -> left_antler_base
    (15, 16), # neck_base -> neck_end
    (17, 18), # throat_base -> throat_end
    (19, 20), # back_base -> back_end
    (19, 21), # back_base -> back_middle
    (20, 21), # back_end -> back_middle
    (22, 23), # tail_base -> tail_end
    (24, 25), # front_left_thai -> front_left_knee
    (25, 26), # front_left_knee -> front_left_paw
    (27, 28), # front_right_thai -> front_right_paw
    (28, 29), # front_right_paw -> front_right_knee
    (30, 31), # back_left_knee -> back_left_paw
    (32, 33), # back_left_thai -> back_right_thai
    (32, 30), # back_left_thai -> back_left_knee
    (33, 34), # back_right_thai -> back_right_paw
    (33, 35), # back_right_thai -> back_right_knee
    (36, 37), # belly_bottom -> body_middle_right
    (37, 38)  # body_middle_right -> body_middle_left
]


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