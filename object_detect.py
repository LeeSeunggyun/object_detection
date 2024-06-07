import math
import os
import cv2
# pip install opencv-python
from ultralytics import YOLO
# pip install ultralytics
import torch
import shutil



# model_test/src 폴더에 있는 모든 이미지들을 models 폴더의 모든 모델로 검사
# detect() 


# 이미지 파일 하나를 모든 모델로 검사
# detect_with_image("model_test/src/KakaoTalk_20240602_225259258.jpg")
# detect_with_image("model_test/src/KakaoTalk_20240602_225259258_01.jpg")
# detect_with_image("model_test/src/KakaoTalk_20240602_230702558.jpg")
# detect_with_image("model_test/src/KakaoTalk_20240602_230702558_01.jpg")


# 이미지 파일 하나를 하나의 모델로 검사
# detect_indiviaully(input_image_path = "model_test/src/KakaoTalk_20240602_225259258.jpg", 
#                 model_output_folder = "model_test/individual_test", 
#                 model_path = "models/best137.pt")



def image_detection(image_path, save_path, model, classNames, targets):
    #--------------------------------추론----------------------------------#
    # 이미지 불러오기
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))
    # 추론 실행
    if(torch.cuda.is_available()):  # CUDA 지원하는 gpu(Nvidia gpu)있는 경우 gpu 사용
        results = model.predict(img, device=0)
    else:
        results = model.predict(img, device='cpu')
    #----------------------------------------------------------------------#
    
    target_count = 0    # number of target plastic bottle
    
    # 추론 결과 시각화
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}_{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            if class_name in targets:   # target plastic bottle
                target_count+=1
                color = (222, 82, 175)
            # elif class_name == "4": # other plastic bottle
            #     color = (0, 149, 255)
            else:                   # not plastic bottle
                color = (85, 45, 255)
            # if conf > 0.25:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
            # Adjust text location if it goes out of frame
            if y1 - t_size[1] < 0:
                text_y = y1 + t_size[1] + 3
            else:
                text_y = y1 - 2
            cv2.putText(img, label, (x1, text_y), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    print(f"target count : {target_count}")
    result_image_path = os.path.join(save_path, os.path.basename(image_path))
    
    # 추론 결과 저장
    cv2.imwrite(result_image_path, img)

    return result_image_path


def detect():
    # 모델들 있는 경로
    model_path = "./models"
    # 검사할 이미지 있는 경로
    input_folder = "./model_test/src"
    # 검사 결과들 저장할 경로
    output_folder = "./model_test"

    # Create a directory for each model
    for model_name in os.listdir(model_path):
        model_output_folder = os.path.join(output_folder, model_name)
        if not os.path.exists(model_output_folder):
            os.makedirs(model_output_folder)

        print(model_name)
        if model_name in ["best137.pt"]:
            classNames = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28']
            targets=['2','4','6','8','10','12']
        elif model_name in ["best232.pt", "best232_3080.pt"]:
            classNames = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
            targets=['0']
        elif model_name in ["best307.pt"]:
            classNames = ['0','1','2','3','4','5','6','7','8']
            targets=['3']
        elif model_name in ["best232_reducedClass.pt"]:
            classNames = ['0','1','2','3','4','5','6','7','8','9']
            targets=['0']
        else:
            print("Invalid model name")
            # Remove the model output folder
            shutil.rmtree(model_output_folder)
            continue
            
        model = YOLO(os.path.join(model_path, model_name))
        for filename in os.listdir(input_folder):
            if filename.endswith(".jpg"):
                input_image_path = os.path.join(input_folder, filename)
                # 추론 수행 + 결과 저장
                result_image_path = image_detection(input_image_path, 
                                                    model_output_folder,
                                                    model,
                                                    classNames,
                                                    targets)

def detect_indiviaully(input_image_path, model_output_folder, model_path):

    # get model name if model_name is from of path
    if model_path.find("/") != -1:
        model_name = model_path.split("/")[-1]

    if model_name in ["best137.pt"]:
        classNames = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28']
        targets=['2','4','6','8','10','12']
    elif model_name in ["best232.pt", "best232_3080.pt"]:
        classNames = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
        targets=['0']
    elif model_name in ["best307.pt"]:
        classNames = ['0','1','2','3','4','5','6','7','8']
        targets=['3']
    elif model_name in ["best232_reducedClass.pt"]:
        classNames = ['0','1','2','3','4','5','6','7','8','9']
        targets=['0']
    else:
        print("Invalid model name")
        
    model = YOLO(model_name)
    result_image_path = image_detection(input_image_path, 
                                        model_output_folder,
                                        model,
                                        classNames,
                                        targets)

# image detection with each models with image file only
def detect_with_image(input_image_path):
    # 모델들 있는 경로
    model_path = "./models"
    # 검사 결과들 저장할 경로
    output_folder = "./model_test"

    # Create a directory for each model
    for model_name in os.listdir(model_path):
        model_output_folder = os.path.join(output_folder, model_name)
        if not os.path.exists(model_output_folder):
            os.makedirs(model_output_folder)

        print(model_name)
        if model_name in ["best137.pt"]:
            classNames = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28']
            targets=['2','4','6','8','10','12']
        elif model_name in ["best232.pt", "best232_3080.pt"]:
            classNames = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
            targets=['0']
        elif model_name in ["best307.pt"]:
            classNames = ['0','1','2','3','4','5','6','7','8']
            targets=['3']
        elif model_name in ["best232_reducedClass.pt"]:
            classNames = ['0','1','2','3','4','5','6','7','8','9']
            targets=['0']
        else:
            print("Invalid model name")
            continue
            
        model = YOLO(os.path.join(model_path, model_name))
        # 추론 수행 + 결과 저장
        result_image_path = image_detection(input_image_path, 
                                            model_output_folder,
                                            model,
                                            classNames,
                                            targets)

