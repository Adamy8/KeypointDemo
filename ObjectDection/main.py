import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("test.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("distance_calculation.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init distance-calculation obj
distance = solutions.DistanceCalculation(model="yolo11n.pt", show=True)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = distance.calculate(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()