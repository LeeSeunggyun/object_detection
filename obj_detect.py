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
