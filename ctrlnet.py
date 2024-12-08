from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, StableDiffusionControlNetPipeline
from diffusers.utils import load_image, make_image_grid
import numpy as np
import torch
from PIL import Image
import cv2
import argparse
import rembg
import os


parser = argparse.ArgumentParser()
parser.add_argument('input_path', type=str)
parser.add_argument('--pp', type=str)
parser.add_argument('--np', type=str) 
parser.add_argument('--scale', type=float) 
args = parser.parse_args()


controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd", use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                            controlnet=controlnet, 
                                                            use_safetensors=True,
                                                            safety_checker = None,
                                                            ).to("cuda")

# controlnet = ControlNetModel.from_pretrained(
#     "diffusers/controlnet-canny-sdxl-1.0",
#     torch_dtype=torch.float16,
#     use_safetensors=True
# )
# vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
# pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     controlnet=controlnet,
#     vae=vae,
#     torch_dtype=torch.float16,
#     use_safetensors=True
# )
# pipe.enable_model_cpu_offload()




input_files = [args.input_path]
input_image = Image.open(input_files[0])
res = 800

(W,H) = (input_image.size[0],input_image.size[1])
if W>H:
    input_image = input_image.resize((res, int(res*(H/W))))
else:
    input_image = input_image.resize((int(res*(W/H)), res))

image = np.array(input_image)

low_threshold = 100
high_threshold = 150

image = cv2.Canny(image, low_threshold, high_threshold)

canny_image = Image.fromarray(image)

prompt = args.pp
negative_prompt = args.np


image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    image=canny_image,
    controlnet_conditioning_scale=args.scale,
).images[0]



name = os.path.basename(input_files[0]).split('.')[0]
os.makedirs('sketch_to_image/' + name, exist_ok=True)

image.save(os.path.join('sketch_to_image/', f'{name}/{name}_after.png'))

input_image = Image.open(f'sketch_to_image/{name}/{name}_after.png')
(H,W) = (input_image.size[0],input_image.size[1]) 
image = input_image.resize((int(H/3), int(W/3)))
image.save(os.path.join('sketch_to_image/', f'{name}/{name}_after_resize.png'))

#python ctrlnet.py multi.png --pp="clear, high quality, white furniture, plant, perfect shape" --np="low quality"
