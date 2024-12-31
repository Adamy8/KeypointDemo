from ultralytics import YOLO
import cv2

model_path = "runs/detect/train10/weights/last.pt"
model = YOLO(model_path)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    results = model(frame)

    frame_with_boxes = results[0].plot()  # This adds bounding boxes to the frame

    cv2.imshow("Detection", frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV window
cap.release()
cv2.destroyAllWindows()