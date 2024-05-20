import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

# JSON 라벨링 파일 경로와 이미지 파일 경로
json_file_path = "880770@5_03001_220913_P1_T1.json"  # 실제 라벨링 파일 경로로 바꿔주세요
image_file_path = "880770@5_03001_220913_P1_T1.jpg"    # 실제 이미지 파일 경로로 바꿔주세요

# JSON 파일 로드
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 이미지 파일 로드
image = cv2.imread(image_file_path)

if image is None:
    print(f"Error loading image: {image_file_path}")
else:
    # ANNOTATION_INFO에서 다각형 좌표 가져오기 및 그리기
    for annotation in data["ANNOTATION_INFO"]:
        points = np.array(annotation["POINTS"], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=10)

    # 이미지를 Matplotlib를 사용하여 표시
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # 축을 숨기기
    plt.show()
