import cv2

from ultralytics import solutions

cap = cv2.VideoCapture(1)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("distance_calculation.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init distance-calculation obj
distance = solutions.DistanceCalculation(model="yolo11n.pt", show=True)

# Example to see the distance
# Assuming `distance.calculate` returns a dictionary or similar structure with distance information
# test_frame = cap.read()[1]  # Read a single frame for testing
# if test_frame is not None:
#     result = distance.calculate(test_frame)
#     print("Distance calculation result:", result)
# else:
#     print("Failed to read a test frame for distance calculation.")

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    
    im0 = distance.calculate(im0)
    

    video_writer.write(im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting video processing loop.")
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()



# # Process video
# while cap.isOpened():
#     success, im0 = cap.read()
#     if not success:
#         print("Video frame is empty or video processing has been successfully completed.")
#         break

#     # Run distance calculation
#     results = distance.calculate(im0)

#     # Draw and print distance next to bounding boxes
#     if results:
#         for det, dist in zip(distance.detections, distance.distances):
#             if det is not None and len(det) >= 4:
#                 x1, y1, x2, y2 = map(int, det[:4])
#                 person_distance = round(dist, 2)

#                 print(f"Person at distance: {person_distance} meters")

#                 # Draw distance on the frame
#                 cv2.putText(im0, f"{person_distance}m", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#                 cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     video_writer.write(im0)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("Exiting video processing loop.")
#         break

# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()