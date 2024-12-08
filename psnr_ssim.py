import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# PSNR 계산 함수
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf') 
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

# SSIM 계산 함수
def calculate_ssim(img1, img2):
    ssim_value, _ = ssim(img1, img2, full=True, multichannel=True)
    return ssim_value

objlist = ["plant", "vase", "bottle2", "cup3", "laptop", "chair1", "monitor1", "stool2", "table5", "sofa4"]

for idx, obj in enumerate(objlist):
    print("========================\n", idx, " computing : ", obj)

    input_file1 = f"eval/{obj}.png"
    input_file2 = f"eval/{obj}_rendered.png"

    img1 = cv2.imread(input_file1, cv2.IMREAD_COLOR)  
    img2 = cv2.imread(input_file2, cv2.IMREAD_COLOR)  

    if img1 is None or img2 is None:
        print(f"Image not found: {input_file1} or {input_file2}")
        continue


    if img1.shape != img2.shape:
        print(f"Resizing {input_file2} to match {input_file1}")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))


    psnr_value = calculate_psnr(img1, img2)
    ssim_value = calculate_ssim(img1, img2)


    print(f"PSNR: {psnr_value:.2f}")
    print(f"SSIM: {ssim_value:.4f}")
