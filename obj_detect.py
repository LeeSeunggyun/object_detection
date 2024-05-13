from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

classNames = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24']

model = YOLO("best.pt")

# Initialize webcam capture
cap = cv2.VideoCapture(0)  # 0 represents the default webcam

# Check if the webcam is opened successfully
assert cap.isOpened(), "Error accessing webcam"

# Get webcam frame properties
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # Assuming a standard webcam frame rate

line_points = [(20, 400), (w, 400)]  # line or region points
classes_to_count = [2, 4, 6, 8, 10, 12] 

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi",
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps,
                                (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                reg_pts=line_points,
                classes_names=model.names,
                draw_tracks=True,
                line_thickness=2)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Error reading frame from webcam")
        break
    tracks = model.track(im0, persist=True, show=False,
                        classes=classes_to_count)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()



import os
import math
import cv2
import torch

pet_model= YOLO("best.pt")

def image_detection(image_path, save_path):
    # --------------------------------추론----------------------------------#
    # 이미지 불러오기
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))
    # 추론 실행
    if torch.cuda.is_available():  # CUDA 지원하는 gpu(Nvidia gpu)있는 경우 gpu 사용
        results = pet_model.predict(img, device=0)
    else:
        results = pet_model.predict(img, device="cpu")
    # ----------------------------------------------------------------------#

    target_count = 0  # number of target plastic bottle

    # 추론 결과 시각화
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f"{class_name}_{conf}"
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            if class_name in ["2", "4", "6", "8", "10", "12"]:  # target plastic bottle
                target_count += 1
                color = (222, 82, 175)
            # elif class_name == "4": # other plastic bottle
            #     color = (0, 149, 255)
            else:  # not plastic bottle
                color = (85, 45, 255)
            # if conf > 0.25:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
            # Adjust text location if it goes out of frame
            if y1 - t_size[1] < 0:
                text_y = y1 + t_size[1] + 3
            else:
                text_y = y1 - 2
            cv2.putText(
                img,
                label,
                (x1, text_y),
                0,
                1,
                [255, 255, 255],
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    print(f"target count : {target_count}")
    result_image_path = os.path.join(save_path, os.path.basename(image_path))

    # 추론 결과 저장
    cv2.imwrite(result_image_path, img)

    return result_image_path