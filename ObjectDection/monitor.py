# distance calculator between two objects (w/ dashboard playground)



import mss
import numpy as np
import cv2
from ultralytics import YOLO
import json
from pynput.keyboard import Key, Controller

# Initialize keyboard controller
keyboard = Controller()

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Load your custom trained YOLOv8 model
model = YOLO('path/to/your/custom_model.pt')

# Set up mss for screen capture
with mss.mss() as sct:
    monitor = {"top": 160, "left": 160, "width": 640, "height": 480}  # Define your monitor region to capture

    while True:  # Continuous loop for real-time processing
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)

        # Convert from BGR to RGB (mss captures in BGR format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run prediction
        results = model.predict(frame_rgb)

        # Extract centroids and class IDs
        centroids = []
        class_ids = []
        for det in results[0].boxes.xyxy:
            if len(det) == 4:  # Ensure that the detection has 4 values
                x1, y1, x2, y2 = det
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                centroids.append(centroid)
                class_ids.append(int(results[0].boxes.cls))  # Convert class tensor to int

        # Calculate distances if there are at least 2 detections
        if len(centroids) > 1:
            distances = np.full((len(centroids), len(centroids)), np.inf)
            for i, point1 in enumerate(centroids):
                for j, point2 in enumerate(centroids):
                    if i != j:
                        distances[i][j] = euclidean_distance(point1, point2)

            # Find the pair with the shortest distance
            i, j = np.unravel_index(distances.argmin(), distances.shape)
            shortest_distance = distances[i][j]

            # Press keys corresponding to the shortest distance pair
            if shortest_distance < threshold_distance:  # Define a threshold distance
                keyboard.press(str(class_ids[i]))  # Press key for object i
                keyboard.release(str(class_ids[i]))
                keyboard.press(str(class_ids[j]))  # Press key for object j
                keyboard.release(str(class_ids[j]))

        # Optionally, save results to JSON
        json_results = results[0].tojson()
        with open('results.json', 'w') as f:
            json.dump(json_results, f)

        # Display the frame with annotations
        annotated_frame = results[0].plot()
        cv2.imshow('YOLOv8 Inference', annotated_frame)
        if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
            break

cv2.destroyAllWindows()