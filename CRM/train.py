"""
training script for imagedream
- the config system is similar with stable diffusion ldm code base(using omigaconf, yaml; target, params initialization, etc.)
- the training code base is similar with unidiffuser training code base using accelerate

"""

from omegaconf import OmegaConf # YAML config system
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import os.path as osp
import numpy as np
import os
import torch
from PIL import Image
import numpy as np
import wandb                    # Tool for logging
from libs.base_utils import get_data_generator, PrintContext
from libs.base_utils import (
    setup,
    instantiate_from_config,
    dct2str,
    add_prefix,
    get_obj_from_str,
)
from absl import logging
from einops import rearrange
from imagedream.camera_utils import get_camera
from libs.sample import ImageDreamDiffusion # Image gererative model
from rich import print


def train(config, unk):
    # using pipeline to extract models
    accelerator, device = setup(config, unk) # accelerator: object to manage multi-gpu training, device: device to run the model
    with PrintContext(f"{'access STAT':-^50}", accelerator.is_main_process):
        print(accelerator.state)
    dtype = {           # data type for the model
        "fp16": torch.float16,
        "fp32": torch.float32,
        "no": torch.float32,
        "bf16": torch.bfloat16,
    }[accelerator.state.mixed_precision]

    num_frames = config.num_frames

    ################## load models ##################
    model_config = config.models.config
    model_config = OmegaConf.load(model_config)
    model = instantiate_from_config(model_config.model) # instantiate the model from the config file
    state_dict = torch.load(config.models.resume, map_location="cpu") # load the model from the checkpoint

    print(model.load_state_dict(state_dict, strict=False))
    print("loaded model from {}".format(config.models.resume))

    latest_step = 0
    if config.get("resume", False):        # resume training from the last checkpoint
        print("resuming from specified workdir")
        ckpts = os.listdir(config.ckpt_root)
        if len(ckpts) == 0:
            print("no ckpt found")
        else:
            latest_ckpt = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))[-1] # get the latest checkpoint
            latest_step = int(latest_ckpt.split("-")[-1])
            print("loadding ckpt from ", osp.join(config.ckpt_root, latest_ckpt))
            unet_state_dict = torch.load( # load the unet model from the checkpoint
                osp.join(config.ckpt_root, latest_ckpt), map_location="cpu"
            )
            print(model.model.load_state_dict(unet_state_dict, strict=False))

    elif config.models.get("resume_unet", None) is not None:
        unet_state_dict = torch.load(config.models.resume_unet, map_location="cpu")
        print(model.model.load_state_dict(unet_state_dict, strict=False)) # strict=False: ignore the mismatched keys
        print(f"______ load unet from {config.models.resume_unet} ______")
    model.to(device)
    model.device = device
    model.clip_model.device = device

    ################# setup optimizer #################
    from torch.optim import AdamW
    from accelerate.utils import DummyOptim

    optimizer_cls = ( # optimizer for the model
        AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(model.model.parameters(), **config.optimizer) # instantiate the optimizer from the config file

    ################# prepare datasets #################
    dataset = instantiate_from_config(config.train_data) # instantiate the dataset from the config file
    eval_dataset = instantiate_from_config(config.eval_data) # instantiate the eval dataset from the config file
    in_the_wild_images = ( # instantiate the in the wild images from the config file
        instantiate_from_config(config.in_the_wild_images)
        if config.get("in_the_wild_images", None) is not None
        else None
    )

    dl_config = config.dataloader
    dataloader = DataLoader(dataset, **dl_config, batch_size=config.batch_size)

    (
        model,
        optimizer,
        dataloader,
    ) = accelerator.prepare(model, optimizer, dataloader)  # prepare the model, optimizer, dataloader for the training

    generator = get_data_generator(dataloader, accelerator.is_main_process, "train") # get the data generator for the training
    if config.get("sampler", None) is not None:
        sampler_cls = get_obj_from_str(config.sampler.target) # get the sampler class from the config file
        sampler = sampler_cls(model, device, dtype, **config.sampler.params)
    else:
        sampler = ImageDreamDiffusion( # instantiate the image dream diffusion from the config file
            model,
            mode=config.mode, # mode for the image dream diffusion
            num_frames=num_frames,
            device=device,
            dtype=dtype,
            camera_views=dataset.camera_views, # camera views option
            offset_noise=config.get("offset_noise", False), # offset noise option
            ref_position=dataset.ref_position,
            random_background=dataset.random_background, # random background option
            resize_rate=dataset.resize_rate, # resize rate: resize the image
        )

    ################# evaluation code #################
    def evaluation():
        return_ls = []
        for i in range(
            accelerator.process_index, len(eval_dataset), accelerator.num_processes # iterate over the eval dataset
        ):
            cond = eval_dataset[i]["cond"] # get the condition from the eval dataset

            images = sampler.diffuse("3D assets.", cond, n_test=2) # diffuse the images using the image dream diffusion
            images = np.concatenate(images, 0) # concatenate the images to the numpy array
            images = [Image.fromarray(images)]
            return_ls.append(dict(images=images, ident=eval_dataset[i]["ident"]))
        return return_ls

    def evaluation2():
        # eval for common used in the wild image
        return_ls = []
        in_the_wild_images.init_item()
        for i in range(
            accelerator.process_index,
            len(in_the_wild_images),
            accelerator.num_processes,
        ):
            cond = in_the_wild_images[i]["cond"] # get the condition from the in the wild images
            images = sampler.diffuse("3D assets.", cond, n_test=2) # diffuse the images using the image dream diffusion
            images = np.concatenate(images, 0) # concatenate the images to the numpy array
            images = [Image.fromarray(images)]
            return_ls.append(dict(images=images, ident=in_the_wild_images[i]["ident"]))
        return return_ls

    if latest_step == 0: # initialize the steps for the training 
        global_step = 0
        total_step = 0
        log_step = 0
        eval_step = 0
        save_step = 0
    else: # resume the steps for the training
        global_step = latest_step // config.total_batch_size
        total_step = latest_step
        log_step = latest_step + config.log_interval
        eval_step = latest_step + config.eval_interval
        save_step = latest_step + config.save_interval

    unet = model.model
    while True:
        item = next(generator) # get the next item from the generator
        unet.train()
        bs = item["clip_cond"].shape[0] # batch size
        BS = bs * num_frames            # total batch size
        item["clip_cond"] = item["clip_cond"].to(device).to(dtype)
        item["vae_cond"] = item["vae_cond"].to(device).to(dtype)
        camera_input = item["cameras"].to(device)
        camera_input = camera_input.reshape((BS, camera_input.shape[-1]))

        gd_type = config.get("gd_type", "pixel") # get the gradient descent type
        if gd_type == "pixel":
            item["target_images_vae"] = item["target_images_vae"].to(device).to(dtype)
            gd = item["target_images_vae"]
        elif gd_type == "xyz":
            item["target_images_xyz_vae"] = (
                item["target_images_xyz_vae"].to(device).to(dtype)
            )
            gd = item["target_images_xyz_vae"]
        elif gd_type == "fusechannel":
            item["target_images_vae"] = item["target_images_vae"].to(device).to(dtype)
            item["target_images_xyz_vae"] = (
                item["target_images_xyz_vae"].to(device).to(dtype)
            )
            gd = torch.cat(
                (item["target_images_vae"], item["target_images_xyz_vae"]), dim=0
            )
        else: # raise error if the gradient descent type is not supported
            raise NotImplementedError

        with torch.no_grad(), accelerator.autocast("cuda"):
            ip_embed = model.clip_model.encode_image_with_transformer(item["clip_cond"]) # encode the image with transformer
            ip_ = ip_embed.repeat_interleave(num_frames, dim=0) # repeat the image embedding

            ip_img = model.get_first_stage_encoding( # get the first stage encoding
                model.encode_first_stage(item["vae_cond"])
            )

            gd = rearrange(gd, "B F C H W -> (B F) C H W") # rearrange the gradient descent

            latent_target_images = model.get_first_stage_encoding(
                model.encode_first_stage(gd)
            )

            if gd_type == "fusechannel": # fuse the channel
                latent_target_images = rearrange(
                    latent_target_images, "(B F) C H W -> B F C H W", B=bs * 2
                )
                image_latent, xyz_latent = torch.chunk(latent_target_images, 2)
                fused_channel_latent = torch.cat((image_latent, xyz_latent), dim=-3)
                latent_target_images = rearrange(
                    fused_channel_latent, "B F C H W -> (B F) C H W"
                )

            if item.get("captions", None) is not None: # get the captions
                caption_ls = np.array(item["caption"]).T.reshape((-1, BS)).squeeze()
                prompt_cond = model.get_learned_conditioning(caption_ls)
            elif item.get("caption", None) is not None:
                prompt_cond = model.get_learned_conditioning(item["caption"])
                prompt_cond = prompt_cond.repeat_interleave(num_frames, dim=0)
            else:
                prompt_cond = model.get_learned_conditioning(["3D assets."]).repeat(
                    BS, 1, 1
                )
            condition = {
                "context": prompt_cond,
                "ip": ip_,
                "ip_img": ip_img,
                "camera": camera_input,
            }

        with torch.autocast("cuda"), accelerator.accumulate(model):
            time_steps = torch.randint(0, model.num_timesteps, (BS,), device=device) # get the time steps
            noise = torch.randn_like(latent_target_images, device=device) # get the noise
            # noise_img, _ = torch.chunk(noise, 2, dim=1)
            # noise = torch.cat((noise_img, noise_img), dim=1)
            x_noisy = model.q_sample(latent_target_images, time_steps, noise) # sample the latent target images
            output = unet(x_noisy, time_steps, **condition, num_frames=num_frames)
            reshaped_pred = output.reshape(bs, num_frames, *output.shape[1:]).permute(
                1, 0, 2, 3, 4
            )
            reshaped_noise = noise.reshape(bs, num_frames, *noise.shape[1:]).permute(
                1, 0, 2, 3, 4
            )
            true_pred = reshaped_pred[: num_frames - 1] # get the true prediction
            fake_pred = reshaped_pred[num_frames - 1 :] # get the fake prediction
            true_noise = reshaped_noise[: num_frames - 1] # get the true noise
            fake_noise = reshaped_noise[num_frames - 1 :] # get the fake noise
            loss = (
                torch.nn.functional.mse_loss(true_noise, true_pred)
                + torch.nn.functional.mse_loss(fake_noise, fake_pred) * 0
            )

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        total_step = global_step * config.total_batch_size # calculate the total step
        if total_step > log_step:
            metrics = dict(
                loss=accelerator.gather(loss.detach().mean()).mean().item(),
                scale=(
                    accelerator.scaler.get_scale()
                    if accelerator.scaler is not None
                    else -1
                ),
            )
            log_step += config.log_interval
            if accelerator.is_main_process:
                logging.info(dct2str(dict(step=total_step, **metrics)))
                wandb.log(add_prefix(metrics, "train"), step=total_step)

        if total_step > save_step and accelerator.is_main_process: # save the model
            logging.info("saving done")
            torch.save(
                unet.state_dict(), osp.join(config.ckpt_root, f"unet-{total_step}")
            )
            save_step += config.save_interval
            logging.info("save done")

        if total_step > eval_step: # evaluate the model
            logging.info("evaluationing")
            unet.eval()
            return_ls = evaluation() # get the evaluation of dataset
            cur_eval_base = osp.join(config.eval_root, f"{total_step:07d}") # get the current evaluation base
            os.makedirs(cur_eval_base, exist_ok=True)
            for item in return_ls:
                for i, im in enumerate(item["images"]):
                    im.save(
                        osp.join(
                            cur_eval_base,
                            f"{item['ident']}-{i:03d}-{accelerator.process_index}-.png",
                        )
                    )

            return_ls2 = evaluation2() # get the evaluation of in the wild images
            cur_eval_base = osp.join(config.eval_root2, f"{total_step:07d}")
            os.makedirs(cur_eval_base, exist_ok=True)
            for item in return_ls2:
                for i, im in enumerate(item["images"]):
                    im.save(
                        osp.join(
                            cur_eval_base,
                            f"{item['ident']}-{i:03d}-{accelerator.process_index}-inthewild.png",
                        )
                    )
            eval_step += config.eval_interval # update the eval step
            logging.info("evaluation done")

        accelerator.wait_for_everyone()
        if total_step > config.max_step: # break the loop if the total step is greater than the max step
            break


if __name__ == "__main__":
    # load config from config path, then merge with cli args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/nf7_v3_SNR_rd_size_stroke.yaml"
    )
    parser.add_argument(
        "--logdir", type=str, default="train_logs", help="the dir to put logs"
    )
    parser.add_argument(
        "--resume_workdir", type=str, default=None, help="specify to do resume"
    )
    args, unk = parser.parse_known_args()
    print(args, unk)
    config = OmegaConf.load(args.config)
    if args.resume_workdir is not None:
        assert osp.exists(args.resume_workdir), f"{args.resume_workdir} not exists"
        config.config.workdir = args.resume_workdir
        config.config.resume = True
    OmegaConf.set_struct(config, True)  # prevent adding new keys
    cli_conf = OmegaConf.from_cli(unk)
    config = OmegaConf.merge(config, cli_conf)
    config = config.config
    OmegaConf.set_struct(config, False)
    config.logdir = args.logdir
    config.config_name = Path(args.config).stem

    train(config, unk)
