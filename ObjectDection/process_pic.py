### Distance Calculation ###
from ultralytics import YOLO
from ultralytics.solutions import distance_calculation
import cv2

model = YOLO("yolo11n.pt")
# model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(1)
assert cap.isOpened(), "Error opening video file"

dist_obj = distance_calculation.DistanceCalculation()

# Process video frames
while True:
    success, frame = cap.read()
    if not success:
        break

    tracks = model.track(frame, persist=True)

    frame = dist_obj.start_process(frame, tracks)

    cv2.imshow('Distance Calculation', frame)
    if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()