import cv2
import os
from tqdm import tqdm
import numpy as np

def combine_images(base_folder, src_path, output_folder_path, model_count):
    # parameters:
    # base_folder: model_test folder
    # src_path: the folder containing the base images
    # output_folder_path: the folder to save the combined images
    # model_count: the number of models to combine
    
    for image_name in tqdm(os.listdir(src_path)):
        
        # Open base image
        if image_name.endswith(".jpg"):
            image_name = image_name.split(".")[0]
            # Open the image
            base_image = cv2.imread(f"{src_path}/{image_name}.jpg")
            
            # Create a new blank image with the desired dimensions
            # the number of max width is 2, 
            # so if the model_count is 6, the grid will be 2x3
            # and if the model_count is 7, the grid will be 2x4
            if(model_count % 2 == 0):
                grid_width = base_image.shape[1] * 2
                grid_height = base_image.shape[0] * (model_count // 2)
            else:
                grid_width = base_image.shape[1] * 2
                grid_height = base_image.shape[0] * ((model_count // 2) + 1)
            grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            # Paste the base image into the grid
            grid_image[0:base_image.shape[0], 0:base_image.shape[1]] = base_image
            
            # for each directory in model_test folder that ends with .pt
            pos_cal = 1
            model_name_list = []
            pos_list = []
            for model_name in os.listdir(base_folder):
                if model_name.endswith(".pt"):
                    # Open the image
                    image = cv2.imread(f"{base_folder}/{model_name}/{image_name}.jpg")
                    # Resize the image to match the base image
                    image = cv2.resize(image, (base_image.shape[1], base_image.shape[0]))
                    
                    # Paste the image into the grid
                    # Calculate the position based on pos_cal
                    if pos_cal % 2 == 1:
                        x = base_image.shape[1]
                    else:
                        x = 0
                    y = base_image.shape[0] * (pos_cal // 2)
                    model_name_list.append(model_name)
                    pos_list.append((x, y))
                    grid_image[y:y+base_image.shape[0], x:x+base_image.shape[1]] = image
                    
                    pos_cal += 1
            
            # Draw the model names on the grid
            # Draw the model names on the grid
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(len(model_name_list)):
                # Draw a filled black rectangle as background for the text
                text_size = cv2.getTextSize(model_name_list[i], font, 5, 10)[0]
                cv2.rectangle(grid_image, 
                            (pos_list[i][0], pos_list[i][1] + 10), 
                            (pos_list[i][0] + text_size[0], pos_list[i][1] + text_size[1] + 10), 
                            (0, 0, 0), 
                            -1)
                # Draw the text
                cv2.putText(grid_image, 
                            model_name_list[i], 
                            (pos_list[i][0], pos_list[i][1] + text_size[1] + 10), 
                            font, 
                            5, 
                            (255, 255, 255), 
                            10, 
                            cv2.LINE_AA)
            
                
        # Save the combined image to a specific location
        cv2.imwrite(f"{output_folder_path}/{image_name}_combined.jpg", grid_image)

combine_images(
    base_folder='model_test',
    src_path='model_test/src',
    output_folder_path='model_test/combined',
    model_count=6
)
