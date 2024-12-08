import os
import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_path', type=str)
parser.add_argument('conf', type=float)
args = parser.parse_args()

model = YOLO('yolov10x.pt')
CHECKPOINT_PATH = os.path.join(os.getcwd(), "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)



input_file = args.input_path

img = cv2.imread(input_file, cv2.IMREAD_UNCHANGED)
if img.shape[2] == 4:
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


(H,W) = (img.shape[0],img.shape[1])

detections = model.predict(img, conf=args.conf)


mask_predictor = SamPredictor(sam)

transformed_boxes = mask_predictor.transform.apply_boxes_torch(
    detections[0].boxes.xyxy, list((W,H))
)


class_ids = detections[0].boxes.cls.cpu().numpy().astype(int).tolist()
class_names=[]
for i in class_ids:
    class_names.append(model.names[i])

mask_predictor.set_image(img)

try:
    masks, _, _ = mask_predictor.predict_torch(
        boxes=transformed_boxes,
        multimask_output=False,
        point_coords=None,
        point_labels=None
    )


    name = os.path.basename(input_file).split('.')[0]
    os.makedirs('segmentation_results/' + name, exist_ok=True)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for idx, mask in enumerate(masks):

        # mask

        segmentation = mask.cpu()

        segmentation_rgb = np.stack((segmentation[0],)*3, axis=-1)

        overlay = np.where(segmentation_rgb, (0, 255, 0), img)

        cv2.imwrite(f'segmentation_results/{name}/obj_{idx}_{class_names[idx]}_mask.png', overlay)


        # object

        height, width, channels = img.shape
        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_image[:, :, :3] = img
        rgba_image[:, :, 3] = segmentation.numpy()[0].astype(np.uint8) * 255


        cv2.imwrite(f'segmentation_results/{name}/obj_{idx}_{class_names[idx]}_rgba.png', rgba_image)

except: 
    print('error')
    exit()
    
#https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints