import cv2
from ultralytics import YOLO
# numpy used as numpy

model_path = "runs/pose/train8/weights/last.pt"
model = YOLO(model_path)

# video_path = "testSample/kubo_video.mp4"
# output_path = "testSample/testResult.mp4"

#---realCase is gitignored---
video_path = "realCase/central_park.mp4"
output_path = "realCase/cp_result.mp4"

bbox_conf_level = 0.5

# skeleton
keypoint_connections = [
    (0, 1), (1, 2), (3, 4), (5, 3), (5, 4), (6, 5), (7, 6), (8, 7), 
    (9, 8), (10, 3), (10, 4), (11, 10), (12, 11), (13, 12), (14, 13),
    (15, 16), (17, 18), (19, 20), (19, 21), (20, 21), (22, 23), 
    (24, 25), (25, 26), (27, 28), (28, 29), (30, 31), (32, 33), 
    (32, 30), (33, 34), (33, 35), (36, 37), (37, 38)
]
# Keypoint connections: 0->1: nose->upper_jaw, 1->2: upper_jaw->lower_jaw, 3->4: mouth_end_right->mouth_end_left, 5->3: right_eye->mouth_end_right, 5->4: right_eye->mouth_end_left, 6->5: right_earbase->right_eye, 7->6: right_earend->right_earbase, 8->7: right_antler_base->right_earend, 9->8: right_antler_end->right_antler_base, 10->3: left_eye->mouth_end_right, 10->4: left_eye->mouth_end_left, 11->10: left_earbase->left_eye, 12->11: left_earend->left_earbase, 13->12: left_antler_base->left_earend, 14->13: left_antler_end->left_antler_base, 15->16: neck_base->neck_end, 17->18: throat_base->throat_end, 19->20: back_base->back_end, 19->21: back_base->back_middle, 20->21: back_end->back_middle, 22->23: tail_base->tail_end, 24->25: front_left_thai->front_left_knee, 25->26: front_left_knee->front_left_paw, 27->28: front_right_thai->front_right_paw, 28->29: front_right_paw->front_right_knee, 30->31: back_left_knee->back_left_paw, 32->33: back_left_thai->back_right_thai, 32->30: back_left_thai->back_left_knee, 33->34: back_right_thai->back_right_paw, 33->35: back_right_thai->back_right_knee, 36->37: belly_bottom->body_middle_right, 37->38: body_middle_right->body_middle_left

ANIMALS = [
    "antelope", "grizzly bear", "killer whale", "beaver", "dalmatian", 
    "persian cat", "horse", "german shepherd", "blue whale", "siamese cat", 
    "skunk", "mole", "tiger", "hippopotamus", "leopard", "moose", 
    "spider monkey", "humpback whale", "elephant", "gorilla", "ox", 
    "fox", "sheep", "seal", "chimpanzee", "hamster", "squirrel", 
    "rhinoceros", "rabbit", "bat", "giraffe", "wolf", "chihuahua", 
    "rat", "weasel", "otter", "buffalo", "zebra", "giant+panda", 
    "deer", "bobcat", "pig", "lion", "mouse", "polar+bear", 
    "collie", "walrus", "raccoon", "cow", "dolphin"
]

def draw_keypoints_and_lines(image, keypoints, connections):
    keypointsFlag = []

    for i in range(39):
        keypoint = keypoints[i]
        x, y, conf = keypoint
        if x == 0 and y == 0:
            keypointsFlag.append(False)
        else:
            keypointsFlag.append(True)

            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
            # 这是写conf（optional）
            # cv2.putText(image, str(round(conf, 3)), (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # draw skeleton
    for start_idx, end_idx in connections:

        if keypointsFlag[start_idx] and keypointsFlag[end_idx]:
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]

            # draw line
            cv2.line(image, 
                        (int(start_point[0]), int(start_point[1])), 
                        (int(end_point[0]), int(end_point[1])), 
                        (0, 0, 255), 2)  # Red
    return image


def draw_bbox(image, bboxes):
    for bbox in bboxes:
        x_min, y_min, x_max, y_max, conf, animal_idx = bbox
        if conf > bbox_conf_level:
            x_min, y_min, x_max, y_max, animal_idx = int(x_min), int(y_min), int(x_max), int(y_max), int(animal_idx)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            animal_name = ANIMALS[animal_idx]
            cv2.putText(image, f"{animal_name} {conf:.2f}", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return image


def process_video(video_path, model, keypoint_connections):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video details (like width, height, and FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Codec for mp4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Reach the End of video.")
            break  # End of video

        # Perform inference using YOLO model (keypoints are part of the output)
        results = model(frame)  # Perform inference

        # print(results)
        # breakpoint()          # for debug


        # for result in results:
        keypoints = results[0].keypoints.cpu().numpy().data[0] # 这 assume one animal，最好做多个
        if len(keypoints) > 0:
            frame_with_lines = draw_keypoints_and_lines(frame.copy(), keypoints, keypoint_connections)
        else:
            frame_with_lines = frame

        # draw bbox
        bboxes = results[0].boxes.cpu().numpy().data
        if len(bboxes) > 0:
            frame_with_boxes = draw_bbox(frame_with_lines, bboxes)
        else:
            frame_with_boxes = frame_with_lines



        # Write the frame to the output video
        out.write(frame_with_boxes)

        # Display the frame (optional)
        cv2.imshow("Animal Pose Estimation", frame_with_lines)

        # Exit if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("pressed q")
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video(video_path, model, keypoint_connections)
