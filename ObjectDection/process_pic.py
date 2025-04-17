### Distance Calculation ###
from ultralytics import YOLO
from ultralytics.solutions import distance_calculation
import cv2

model = YOLO("yolo11n.pt")

image = cv2.imread("pic_person.jpg")
assert image is not None, "Error opening image"

distance = distance_calculation.DistanceCalculation(model="yolo11n.pt", show=True)


if image is not None:
    result = distance.calculate(image)
    print("Distance calculation result:", result)
else:
    print("Failed to read a test frame for distance calculation.")


# cv2.imshow("Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()