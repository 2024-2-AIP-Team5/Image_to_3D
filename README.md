# Towards High-Quality Image-to-3D Generation

### AIP Team 5

#### TEAM NAME : 
비전이동
#### TEAM MEMBERS : 
박진호(2018312612), 김주현(2019314455) , 최준영(2019314833), 박재영(2019310338), 김동현(2019312104), 이호준
(2020312401)

## ✨ 3D reconstruction

- NeRF

  paper : https://arxiv.org/pdf/2003.08934
  
  survey : https://arxiv.org/pdf/2210.00379

- 3DGS

  paper : https://arxiv.org/pdf/2308.04079

  survey : https://arxiv.org/abs/2401.03890

## ✨ Image to 3D

- DreamGaussian (ICLR 2024)

  paper : https://arxiv.org/pdf/2309.16653
  
  github : https://github.com/dreamgaussian/dreamgaussian

- CRM (ECCV 2024)

  paper : https://arxiv.org/pdf/2403.05034
  
  github : https://github.com/thu-ml/CRM
  
- InstantMesh

  paper : https://arxiv.org/pdf/2404.07191
  
  github : https://github.com/TencentARC/InstantMesh


## ✨ Super-resolution

- DRCT (CVPR 2024)

  paper : https://arxiv.org/pdf/2404.00722

- IPG (CVPR 2024)

  paper : https://openaccess.thecvf.com/content/CVPR2024/papers/Tian_Image_Processing_GNN_Breaking_Rigidity_in_Super-Resolution_CVPR_2024_paper.pdf

  
  
# ⚙️ Dependencies and Installation

We recommend using `Python>=3.10`, `PyTorch>=2.1.0`, and `CUDA>=12.1`.
```bash
conda create --name mesh python=3.10 -y
conda activate mesh
pip install -U pip

# Ensure Ninja is installed
conda install Ninja

# Install the correct version of CUDA
conda install cuda -c nvidia/label/cuda-12.1.0

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7

# For Linux users: Install Triton 
pip install triton

# For Windows users: Use the prebuilt version of Triton provided here:
pip install https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl

# Install other requirements
pip install -r requirements.txt

# For huggingface error (optional)
pip install --upgrade huggingface_hub

#for SR
python setup_DRCT.py develop
python setup_IPG.py install
```

Put model path to mesh_sr_finetune/

DRCT (DRCT-L_X4): https://drive.google.com/drive/folders/1QJHdSfo-0eFNb96i8qzMJAPw31u9qZ7U

IPG : https://huggingface.co/yuchuantian/IPG/blob/main/IPG_SRx4.pth

## ✨ Running with command line

To generate 3D meshes from images via command line, simply run:
```bash
python run.py configs/instant-mesh-large.yaml Sample/hatsune_miku.png --save_video 
```

We use [rembg](https://github.com/danielgatis/rembg) to segment the foreground object. If the input image already has an alpha mask, please specify the `no_rembg` flag:
```bash
python run.py configs/instant-mesh-large.yaml Sample/hatsune_miku.png --save_video --no_rembg
```

By default, our script exports a `.obj` mesh with vertex colors, please specify the `--export_texmap` flag if you hope to export a mesh with a texture map instead (this will cost longer time):
```bash
python run.py configs/instant-mesh-large.yaml Sample/hatsune_miku.png --save_video --export_texmap
```

Please use a different `.yaml` config file in the [configs](./configs) directory if you hope to use other reconstruction model variants. For example, using the `instant-nerf-large` model for generation:
```bash
python run.py configs/instant-nerf-large.yaml Sample/hatsune_miku.png --save_video
```

### ✨ sketch to image 

To generate image from sketch, use ctrlnet.py :
```bash
python ctrlnet.py multi_obj_sketch.png --pp="clear, high quality, white furniture, plant, perfect shape" --np="low quality" --scale=1
```



### ✨ image segmentation

We recommend using another environment.

```bash
conda create --name seg python=3.10 -y
conda activate seg
pip install ultralytics
pip install opencv-python-headless
pip install supervision
pip install segment_anything
```

Get **sam_vit_h_4b8939.pth** from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints

Put **sam_vit_h_4b8939.pth** to mesh_sr_finetune/


segementation.py usage:

```bash
python segmentation.py sketch_to_image/multi_obj_sketch/multi_obj_sketch_after.png 0.5
```

### ✨ Sketch to 3D Workflow
demonstration of the process converting sketches into 3D models by utilizing ControlNet and segmentation. The key steps are:

#### 1. Sketch to Image Conversion:
ControlNet is used to convert the input sketch into a detailed image.

#### 2. Segmentation:
The generated image is segmented into different objects to identify objects for 3D modeling.

#### 3. 3D Model Generation:
Each segmented object is then converted into its corresponding 3D model using a instantmesh pipeline.

### ✨ Super-resolution

To generate 3D meshes from images with sr, specify --sr :
```bash
python run.py configs/instant-mesh-large.yaml Sample/hatsune_miku.png --save_video --sr DRCT

python run.py configs/instant-mesh-large.yaml Sample/hatsune_miku.png --save_video --sr IPG
```

### ✨ Finetuning

To use finetuning via command line, simply run:
```bash
python ft.py configs/instant-mesh-large.yaml Sample/hatsune_miku.png --save_video
```

For multi images, run bash:
```bash
bash run.sh Sample
```

Open the yaml file and modify the following parameters according to your fine-tuning requirements:

- lr: Learning rate for fine-tuning.
- epochs: Number of epochs to run for fine-tuning.
- StepLR_step_size: The step size for the learning rate scheduler (StepLR).
- StepLR_gamma: Factor by which the learning rate is multiplied at each step.
- one_view / six_views: Choose between one or six views for finetuning.
- mse_loss_weight: Weight for the Mean Squared Error (MSE) loss.
- lpips_loss_weight: Weight for the LPIPS loss (used to measure perceptual similarity).
- flexicubes_reg_loss_weight: Weight for the flexicubes regularization loss.
- save_finetune_result: Whether to save the fine-tuning results.


### ✨ eval 

To get IOU scores from images in eval folder,

just run iou.py :

```bash
python iou.py
```

The results will be like this :

```bash
========================
 0  computing :  plant

0: 640x640 1 potted plant, 1 vase, 20.7ms
Speed: 4.6ms preprocess, 20.7ms inference, 23.5ms postprocess per image at shape (1, 3, 640, 640)

0: 640x640 1 potted plant, 1 vase, 20.8ms
Speed: 4.1ms preprocess, 20.8ms inference, 5.0ms postprocess per image at shape (1, 3, 640, 640)
IOU :  0.9770999009419561
========================
 1  computing :  vase

0: 640x640 1 vase, 20.6ms
Speed: 2.1ms preprocess, 20.6ms inference, 5.0ms postprocess per image at shape (1, 3, 640, 640)

0: 640x640 1 vase, 20.0ms
Speed: 2.2ms preprocess, 20.0ms inference, 5.0ms postprocess per image at shape (1, 3, 640, 640)
IOU :  0.9712203689988091

...
```
To get PSNR/SSIM,

just run psnr_ssim.py :

```bash
python psnr_ssim.py
```

