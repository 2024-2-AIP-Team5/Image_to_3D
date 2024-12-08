
import os
import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO
from PIL import Image

model = YOLO('yolov10x.pt')
# model = YOLO('yolov10b.pt')
# model = YOLO('yolov10l.pt')

CHECKPOINT_PATH = os.path.join(os.getcwd(), "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

objlist = ["plant","vase","bottle2","cup3","laptop","chair1","monitor1","stool2","table5","sofa4"]

for idx,obj in enumerate(objlist):
    
    print("========================\n",idx ," computing : ",obj)    

    input_file1 = f"eval/{obj}.png"
    input_file2 = f"eval/{obj}_rendered.png"

    img1 = cv2.imread(input_file1, cv2.IMREAD_UNCHANGED)
    if img1.shape[2] == 4:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2RGB)

    img2 = cv2.imread(input_file2, cv2.IMREAD_UNCHANGED)
    if img2.shape[2] == 4:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2RGB)
        


    detections1 = model.predict(img1, conf=0.1)
    detections2 = model.predict(img2, conf=0.1)
    
    
    try:
        xyxy1 =  detections1[0].boxes.xyxy
        xyxy2 =  detections2[0].boxes.xyxy
        
        xyxy1[0]
        xyxy2[0]
        
        # change index ( e.g monitor -> xyxy1[1] )
        x1_min, y1_min, x1_max, y1_max = map(float,xyxy1[0])
        x2_min, y2_min, x2_max, y2_max = map(float,xyxy2[0])
        
    except:
        print("no detections")
        
        
    
        print("find manually")
        
        img = cv2.imread(input_file1)
        

        lower_white = np.array([0, 0, 0], dtype=np.uint8)  
        upper_white = np.array([240, 240, 240], dtype=np.uint8)  
        mask = cv2.inRange(img, lower_white, upper_white)  

        non_white_pixels = cv2.findNonZero(mask)  

        if non_white_pixels is not None:

            x_min, y_min = np.min(non_white_pixels[:, 0], axis=0)
            x_max, y_max = np.max(non_white_pixels[:, 0], axis=0)

            print(f"First non-white area bounds: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")


        x1_min, y1_min, x1_max, y1_max = x_min, y_min, x_max, y_max

        img = cv2.imread(input_file2)


        lower_white = np.array([0, 0, 0], dtype=np.uint8)  
        upper_white = np.array([240, 240, 240], dtype=np.uint8)  
        mask = cv2.inRange(img, lower_white, upper_white)  


        non_white_pixels = cv2.findNonZero(mask)  

        if non_white_pixels is not None:
            x_min, y_min = np.min(non_white_pixels[:, 0], axis=0)
            x_max, y_max = np.max(non_white_pixels[:, 0], axis=0)

            print(f"First non-white area bounds: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
            
            
        x2_min, y2_min, x2_max, y2_max = x_min, y_min, x_max, y_max
        

    
    
    
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)


    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)


    intersection_area = inter_width * inter_height
    

    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    


    union_area = bbox1_area + bbox2_area - intersection_area


    iou = intersection_area / union_area if union_area != 0 else 0
    
    print("IOU : ",iou)
