
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# from cuda_check import print_gpu_tensors, print_model_parameters

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

parser.add_argument('--gpu', type=str, default=0)

args = parser.parse_args()
seed_everything(args.seed)

###############################################################################
# Stage 0: Configuration.
###############################################################################

config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

finetuning_config = config.finetuning_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

device = torch.device('cuda')

# load reconstruction model
print('Loading reconstruction model ...')


model = instantiate_from_config(model_config) #lrm_mseh.py
if os.path.exists(infer_config.model_path):
    model_ckpt_path = infer_config.model_path
else:
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
model.load_state_dict(state_dict, strict=True) #set weights as ckpt
model = model.to(device)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=30.0) #src.models.geometry.rep_3d.flexicubes_geometry.FlexiCubesGeometry 
model = model.eval() 

# make output directories
image_path = os.path.join(args.output_path, 'images')
mesh_path = os.path.join(args.output_path, 'meshes')
video_path = os.path.join(args.output_path, 'videos')
os.makedirs(image_path, exist_ok=True)
os.makedirs(mesh_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)

# process input files
if os.path.isdir(args.input_path):
    input_files = [
        os.path.join(args.input_path, file) 
        for file in os.listdir(args.input_path) 
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp')
    ]
else:
    input_files = [args.input_path]
print(f'Total number of input images: {len(input_files)}')



###############################################################################
# Stage 1: Multiview generation.
###############################################################################

name = os.path.basename(input_files[0]).split('.')[0]
os.makedirs(f"{name}", exist_ok=True)

outputs = []

try:
    opened_image = Image.open(os.path.join(image_path, f'{name}.png'))
    print(f"Open image in {os.path.join(image_path, f'{name}.png')}")
    
    images = np.asarray(opened_image, dtype=np.float32) / 255.0        
except:
    
    rembg_session = None if args.no_rembg else rembg.new_session()

    

    print(f'Imagining {name} ...')

    # remove background optionally
    input_image = Image.open(input_files[0])
    if not args.no_rembg:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)
    
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
    
    
    # sampling
    output_image = pipeline(
        input_image, 
        num_inference_steps=args.diffusion_steps, 
    ).images[0]
    
    del pipeline
    
    output_image.save(os.path.join(image_path, f'{name}.png'))
    print(f"Image saved to {os.path.join(image_path, f'{name}.png')}")
    images = np.asarray(output_image, dtype=np.float32) / 255.0        
    
                       
images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

outputs.append({'name': name, 'images': images})
    



#save input image 
import shutil
shutil.copy2(input_files[0], f'{name}/{name}.png')
shutil.copy2(os.path.join(image_path, f'{name}.png'), f'{name}/{name}_zero123_output.png')


#get GT image as 320*320 
rembg_session = None if args.no_rembg else rembg.new_session()
input_image = Image.open(input_files[0])
input_image = remove_background(input_image, rembg_session)
input_image = input_image.resize((320, 320))

background = Image.new('RGBA', (320,320), (255, 255, 255, 255))
background.paste(input_image, (0, 0), input_image)
background.save(f'{name}/{name}_GT.png')

###############################################################################
# Stage 2: Reconstruction.
###############################################################################

input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale).to(device)
chunk_size = 20 if IS_FLEXICUBES else 1



for idx, sample in enumerate(outputs):
    name = sample['name']
    print(f'[{idx+1}/{len(outputs)}] Creating {name} ...')

    images = sample['images'].unsqueeze(0).to(device)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1) #meaningless?? clamping?

    if args.view == 4:
        indices = torch.tensor([0, 2, 4, 5]).long().to(device)
        images = images[:, indices]
        input_cameras = input_cameras[:, indices]

    ################# modified ##################
    
    import camera_util 
    import inspect


    optim_cameras = camera_util.get_optimize_cameras(batch_size=1, radius=4.0).to(device)
    optim_cameras_GT = camera_util.get_optimize_cameras_GT(batch_size=1, radius=4.0).to(device)
    
    if os.path.exists(f'{name}/{name}_enc.pt'):
        #load encoded image
        image_feats = torch.load(f'{name}/{name}_enc.pt')
    else:
        #encode images 
        B = images.shape[0] # B = 6
        image_feats = model.encoder(images, input_cameras)
        image_feats = rearrange(image_feats, '(b v) l d -> b (v l) d', b=B)
        torch.save(image_feats, f'{name}/{name}_enc.pt')
        image_feats = torch.load(f'{name}/{name}_enc.pt')

    if(config_name == "instant-mesh-base"):
        if os.path.exists(f'{name}/{name}_pl_b.pt'):
            #load planes
            planes = torch.load(f'{name}/{name}_pl_b.pt')
        else:
            planes = model.transformer(image_feats)
            torch.save(planes, f'{name}/{name}_pl_b.pt')
            planes = torch.load(f'{name}/{name}_pl_b.pt')

    elif(config_name == "instant-mesh-large"):
        if os.path.exists(f'{name}/{name}_pl_l.pt'):
            #load planes
            planes = torch.load(f'{name}/{name}_pl_l.pt')
        else:
            planes = model.transformer(image_feats)
            torch.save(planes, f'{name}/{name}_pl_l.pt')
            planes = torch.load(f'{name}/{name}_pl_l.pt')

    model.encoder = model.encoder.cpu()
    # model.transformer = model.transformer.cpu()
    del(model.encoder)
    # del(model.transformer)
    torch.cuda.empty_cache()
    
    image_feats.to(device)
    planes.to(device)
    
    import lpips
    loss_fn_vgg = lpips.LPIPS(net='vgg') 
    loss_fn_vgg = loss_fn_vgg.cuda()

# Regulrarization loss for FlexiCubes
def sdf_reg_loss_batch(sdf, all_edges):
    sdf_f1x6x2 = sdf[:, all_edges.reshape(-1)].reshape(sdf.shape[0], -1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = F.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
               F.binary_cross_entropy_with_logits(
                   sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff


l_loss=[]
l_loss_mse=[]
l_loss_lpips=[]
l_loss_reg=[]

render_imgs=[]
residual=[]


def save_res(images, res, name, iter):
    #graph
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 12))

    plt.plot(l_loss, label='Total Loss')
    plt.plot(l_loss_mse, label='MSE Loss')
    plt.plot(l_loss_lpips, label='LPIPS Loss')
    plt.plot(l_loss_reg, label='reg Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{name}/{name}_loss_graph.png')
    
    #make video, input/zero123/rendered/residual
    from torchvision.transforms import ToPILImage
    from PIL import Image, ImageDraw, ImageFont
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as F
    import cv2
    to_pil = F.to_pil_image
    
    a = Image.open(f'{name}/{name}_GT.png')  
    b = to_pil(images[0][0])
    
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default() 
        

    fps = 5

    temp = []
    for i in range(iter):
        c = to_pil(render_imgs[i])
        d = to_pil(residual[i]) 
        
        new_img = Image.new('RGB', (1280, 360), (255,255,255)) 
        new_img.paste(a, (0, 0))  
        new_img.paste(b, (320, 0))  
        new_img.paste(c, (640, 0))  
        new_img.paste(d, (960, 0))  


        draw = ImageDraw.Draw(new_img)
        draw.text((10, 325), f"Iteration {i} input/zero123/rendered/residual", font=font, fill="black")
        
        temp.append(np.array(new_img))

    import imageio
    # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    temp = [(frame).astype(np.uint8) for frame in temp]
    writer = imageio.get_writer(f"{name}.mp4", fps=fps)
    for frame in temp:
        writer.append_data(frame)
    writer.close()    
    
params = [
    # {"params": model.synthesizer.decoder.parameters()},
    # {"params": planes},
    # {"params": model.transformer.parameters()},
    
    {"params": model.transformer.layers[15].parameters()},
    {"params": model.transformer.norm.parameters()},
    {"params": model.transformer.deconv.parameters()},
]      #### transformer 

optim = torch.optim.Adam(params, lr=finetuning_config.lr, eps=1e-15)

from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optim, step_size=finetuning_config.StepLR_step_size, gamma=finetuning_config.StepLR_gamma)

model.train()

images.to(device)


        
def get_mesh_video(step,path_name):
    os.makedirs(f'{name}/results', exist_ok=True)
    
    print("========================================================")
    print(f"saving mesh & video, step = {step}")
    
    with torch.no_grad():
        mesh_path_idx = os.path.join(f'{name}/results/{name}_{step}.obj')

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


    with torch.no_grad():
        if args.save_video:
            video_path_idx = os.path.join(f'{name}/results/{name}_{step}.mp4')
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
            
    print("========================================================")
      


iter = finetuning_config.epochs
for i in range (iter):
    # get triplane
    # planes = model.forward_planes(images, input_cameras)

    # modify upper method : encoder -> decoder (1) or save weights (2)

    planes = model.transformer(image_feats)
    
    render_out = model.forward_geometry(
        planes,
        optim_cameras,
        render_size=320,
    )
    
    frame = render_out["img"]


    loss_mse = ((frame - images) ** 2).mean()

    loss_mse_1view = ((frame[0][0] - images[0][0]) ** 2).mean()

    loss_lpips = 0
    for idx in range(6):
        loss_lpips += loss_fn_vgg(frame[0][idx], images[0][idx]).mean()

    # flexicubes regularization loss
    sdf = render_out['sdf']
    sdf_reg_loss = render_out['sdf_reg_loss']
    sdf_reg_loss_entropy = sdf_reg_loss_batch(sdf, model.geometry.all_edges).mean() * 0.01
    _, flexicubes_surface_reg, flexicubes_weights_reg = sdf_reg_loss
    flexicubes_surface_reg = flexicubes_surface_reg.mean() * 0.5
    flexicubes_weights_reg = flexicubes_weights_reg.mean() * 0.1

    loss_reg = sdf_reg_loss_entropy + flexicubes_surface_reg + flexicubes_weights_reg

    loss_reg *= finetuning_config.flexicubes_reg_loss_weight
    loss_lpips *= finetuning_config.lpips_loss_weight
    loss_mse *= finetuning_config.mse_loss_weight
    loss_mse_1view *= finetuning_config.mse_loss_weight
    
    if finetuning_config.one_view:
        loss = loss_mse_1view + loss_reg + loss_lpips
        l_loss_mse.append(loss_mse_1view.cpu().detach().numpy())
    else:
        loss = loss_mse + loss_reg + loss_lpips
        l_loss_mse.append(loss_mse.cpu().detach().numpy())
    
    loss.backward()
    
    optim.step()
    
    optim.zero_grad()
    
    

    l_loss.append(loss.cpu().detach().numpy())
    
    l_loss_lpips.append(loss_lpips.cpu().detach().numpy())
    l_loss_reg.append(loss_reg.cpu().detach().numpy())
        
    render_imgs.append(frame[0][0])  
    residual.append(images[0][0] - frame[0][0])
    
    if finetuning_config.one_view:
        print(f"{i} / total loss : {loss.item():.5f} / mse loss : {loss_mse_1view.item():.5f} / lpips loss : {loss_lpips.item():.5f} / reg loss : {loss_reg.item():.5f}")
    else:
        print(f"{i} / total loss : {loss.item():.5f} / mse loss : {loss_mse.item():.5f} / lpips loss : {loss_lpips.item():.5f} / reg loss : {loss_reg.item():.5f}")


    if i%10==0:
        get_mesh_video(i,name)
        
    scheduler.step()
    
if finetuning_config.save_finetune_result:
    save_res(images, model.grid_res, name, iter)

#get rendered GT image with fixed elevation & azimuth (0,0) 
render_out = model.forward_geometry(
    planes,
    optim_cameras_GT,
    render_size=320,
)

frame = render_out["img"]

image = Image.fromarray((frame[0][0].permute(1,2,0).detach().cpu().numpy() * 255).astype('uint8'))
image.save(f'{name}/{name}_GT_rendered.png')

# get mesh
with torch.no_grad():
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
with torch.no_grad():
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
