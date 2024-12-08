import os
import sys
import argparse

###############################################################################
# Arguments.
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('input_path', type=str, help='Path to input image or directory.')
parser.add_argument('--output_path', type=str, default='results/', help='Output directory.')
parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
parser.add_argument('--gpus', type=str, default="0", help='gpu ids to use.')
parser.add_argument('--sr', type=str, default='None', choices=["None", "DRCT", "IPG"], help='choose SR model.')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video

#DRCT
import cv2



seed_everything(args.seed)

config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

device = torch.device('cuda')

image_file = args.input_path
name = os.path.basename(image_file).split('.')[0]

# load diffusion model
print('Loading diffusion model ...')
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2", 
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# load custom white-background UNet
print('Loading custom white-background unet ...')
if os.path.exists(infer_config.unet_path):
    unet_ckpt_path = infer_config.unet_path
else:
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device)

# load reconstruction model
print('Loading reconstruction model ...')
model = instantiate_from_config(model_config)
if os.path.exists(infer_config.model_path):
    model_ckpt_path = infer_config.model_path
else:
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
model.load_state_dict(state_dict, strict=True)

model = model.to(device)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()


# make output directories
os.makedirs(args.output_path, exist_ok=True)

sr_path = os.path.join(args.output_path, 'sr')
sr_drct_path = os.path.join(args.output_path, 'sr/DRCT')
sr_ipg_path = os.path.join(args.output_path, 'sr/IPG')
image_path = os.path.join(args.output_path, 'images')
mesh_path = os.path.join(args.output_path, 'meshes')
video_path = os.path.join(args.output_path, 'videos')

os.makedirs(sr_path, exist_ok=True)
os.makedirs(sr_drct_path, exist_ok=True)
os.makedirs(sr_ipg_path, exist_ok=True)
os.makedirs(image_path, exist_ok=True)
os.makedirs(mesh_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)


#SR input image
if args.sr == "DRCT":

    mpath = ""
    for path in sys.path:
        if 'site-packages/basicsr-1.3.4.9-py3.10.egg' in path:
            mpath = path
    sys.path.insert(0, mpath)

    from drct.archs.DRCT_arch import *

    # set up model (DRCT-L)
    model_drct = DRCT(upscale=4, in_chans=3,  img_size= 64, window_size= 16, compress_ratio= 3,squeeze_factor= 30,
                        conv_scale= 0.01, overlap_ratio= 0.5, img_range= 1., depths= [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                        embed_dim= 180, num_heads= [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], gc= 32,
                        mlp_ratio= 2, upsampler= 'pixelshuffle', resi_connection= '1conv')

    model_drct.load_state_dict(torch.load("DRCT-L_X4.pth")['params'], strict=True)
    model_drct.eval()
    model_drct = model_drct.to(device)

    print('Creating', name+'_DRCT_X4')
    # read image
    img = cv2.imread(image_file, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    
    #img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0).to(device)

    window_size = 16
    
    # inference
    try:
        with torch.no_grad():
            _, _, h_old, w_old = img.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
            output = output = model_drct(img)
            output = output[..., :h_old * 4, :w_old * 4]

    except Exception as error:
        print('Error', error, name)
    else:
        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(os.path.join(sr_drct_path, f'{name}_DRCT_X4.png'), output)

    image_file = os.path.join(sr_drct_path, f'{name}_DRCT_X4.png')
    print(f"Image saved to {image_file}")
    name = name + '_DRCT_X4'

if args.sr == "IPG":

    mpath = ""
    for path in sys.path:
        if 'site-packages/basicsr-0.1.0-py3.10.egg' in path:
            mpath = path
    sys.path.insert(0, mpath)

    import shutil

    dest_dir = 'IPG/basicsr/inputs'

    file_name = os.path.basename(image_file)

    dest_file = os.path.join(dest_dir, file_name)

    shutil.copy2(image_file, dest_file)

    print('Creating', name+'_IPG_X4')

    eval_cmd = f"python IPG/basicsr/test.py"
    os.system(eval_cmd)

    image_file = os.path.join(sr_ipg_path, f'{name}_IPG_X4.png')
    shutil.copy2(f"IPG/results/visualization/InferenceDataset/{name}_IPG_X4.png", image_file)
    print(f"Image saved to {image_file}")
    name = name + '_IPG_X4'

    os.remove(dest_file) 


###############################################################################
# Stage 1: Multiview generation.
###############################################################################

rembg_session = None if args.no_rembg else rembg.new_session()

outputs = []

print(f'Imagining {name} ...')

# remove background optionally
input_image = Image.open(image_file)
if not args.no_rembg:
    input_image = remove_background(input_image, rembg_session)
    input_image = resize_foreground(input_image, 0.85)

# sampling
output_image = pipeline(
    input_image, 
    num_inference_steps=args.diffusion_steps, 
).images[0]

output_image.save(os.path.join(image_path, f'{name}.png'))
print(f"Image saved to {os.path.join(image_path, f'{name}.png')}")

images = np.asarray(output_image, dtype=np.float32) / 255.0
images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

outputs.append({'name': name, 'images': images})

# delete pipeline to save memory
del pipeline

#SR zero123 output -> 6 images

image_files = []
if args.sr == "DRCT":

    print('Creating SR zero123 output of', name)

    for idx in range(6):
        img = images[idx].permute(1,2,0).numpy()
        img = img[..., [2, 1, 0]] 
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        
        img = img.unsqueeze(0).to(device)

        window_size = 16
        
        # inference
        try:
            with torch.no_grad():
                _, _, h_old, w_old = img.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
                img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
                output = output = model_drct(img)
                output = output[..., :h_old * 4, :w_old * 4]

        except Exception as error:
            print('Error', error, name)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(sr_drct_path, f'{name}_{idx}.png'), output)

        image_file = os.path.join(sr_drct_path, f'{name}_{idx}.png')
        print(f"Image saved to {image_file}")

        image_files.append(image_file)

    new_size = (320, 320)

    total_width = new_size[0] * 2
    total_height = new_size[1] * 3

    new_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    for i, image_file in enumerate(image_files):
        img = Image.open(image_file)
        img_resized = img.resize(new_size)  

        x_offset = (i % 2) * new_size[0]  
        y_offset = (i // 2) * new_size[1]  
        new_image.paste(img_resized, (x_offset, y_offset))

    new_image.save(os.path.join(image_path, f'{name}_after_DRCT.png'))

    images = Image.open(os.path.join(image_path, f'{name}_after_DRCT.png'))

    images = np.asarray(output_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

    outputs = []
    outputs.append({'name': name, 'images': images})

if args.sr == "IPG":

    print('Creating SR zero123 output of', name)
    import shutil

    for idx in range(6):

        img = images[idx].permute(1, 2, 0).numpy()
        img = Image.fromarray((img * 255).astype('uint8'))
        img.save(f'IPG/basicsr/inputs/{name}_{idx}.png')

        eval_cmd = f"python IPG/basicsr/test.py"
        os.system(eval_cmd)

        image_file = os.path.join(sr_ipg_path, f'{name}_{idx}.png')
        shutil.copy2(f"IPG/results/visualization/InferenceDataset/{name}_{idx}_IPG_X4.png", image_file)
        print(f"Image saved to {image_file}")

        os.remove(f'IPG/basicsr/inputs/{name}_{idx}.png') 

        image_files.append(image_file)

    new_size = (320, 320)

    total_width = new_size[0] * 2
    total_height = new_size[1] * 3

    new_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    for i, image_file in enumerate(image_files):
        img = Image.open(image_file)
        img_resized = img.resize(new_size)  

        x_offset = (i % 2) * new_size[0]  
        y_offset = (i // 2) * new_size[1]  
        new_image.paste(img_resized, (x_offset, y_offset))

    new_image.save(os.path.join(image_path, f'{name}_after_IPG.png'))

    images = Image.open(os.path.join(image_path, f'{name}_after_IPG.png'))

    images = np.asarray(output_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

    outputs = []
    outputs.append({'name': name, 'images': images})
###############################################################################
# Stage 2: Reconstruction.
###############################################################################

def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras

def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    
    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    return frames

input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale).to(device)
chunk_size = 20 if IS_FLEXICUBES else 1

for idx, sample in enumerate(outputs):
    name = sample['name']
    print(f'Creating {name} ...')

    images = sample['images'].unsqueeze(0).to(device)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

    if args.view == 4:
        indices = torch.tensor([0, 2, 4, 5]).long().to(device)
        images = images[:, indices]
        input_cameras = input_cameras[:, indices]

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, input_cameras)

        # get mesh
        mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=args.export_texmap,
            **infer_config,
        )
        if args.export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_path_idx,
            )
        else:
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)
        print(f"Mesh saved to {mesh_path_idx}")

        # get video
        if args.save_video:
            video_path_idx = os.path.join(video_path, f'{name}.mp4')
            render_size = infer_config.render_resolution
            render_cameras = get_render_cameras(
                batch_size=1, 
                M=120, 
                radius=args.distance, 
                elevation=20.0,
                is_flexicubes=IS_FLEXICUBES,
            ).to(device)
            
            frames = render_frames(
                model, 
                planes, 
                render_cameras=render_cameras, 
                render_size=render_size, 
                chunk_size=chunk_size, 
                is_flexicubes=IS_FLEXICUBES,
            )

            save_video(
                frames,
                video_path_idx,
                fps=30,
            )
            print(f"Video saved to {video_path_idx}")
