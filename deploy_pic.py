from ultralytics import YOLO
import cv2

model_path = "runs/pose/train8/weights/last.pt"
model = YOLO(model_path)

image_path = 'testSample/davisHorse.jpg'

img = cv2.imread(image_path)

model = YOLO(model_path)

results = model(image_path)[0]

for result in results:
    keypoints = result.keypoints.cpu().numpy().data[0]
    # print(keypoints)
    # breakpoint()

    if keypoints is not None:
        for i in range(39):
            keypoint = keypoints[i]
            x, y, conf = keypoint
            if conf > 0:
                cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(img, str(i), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()