import torch
import tensorflow as tf
import os
import logging
import tqdm
import numpy as np
from torchvision.utils import save_image, make_grid
import PIL
from PIL import Image


def restore_checkpoint(ckpt_dir, state, device):
    if not tf.io.gfile.exists(ckpt_dir):
        tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
        logging.warning(
            f"No checkpoint found at {ckpt_dir}. " f"Returned the same state as input"
        )
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state["optimizer"].load_state_dict(loaded_state["optimizer"])
        state["model"].load_state_dict(loaded_state["model"], strict=False)
        state["ema"].load_state_dict(loaded_state["ema"])
        state["step"] = loaded_state["step"]
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
    }
    torch.save(saved_state, ckpt_dir)


def save_gif(all_samples, config, ckpt, workdir, image_dir='/images'):
    """
    Save samples as a GIF file.
    
    Args:  
      - all_samples: list of samples from the model
      - config: configuration file
      - ckpt: checkpoint number/name, string or int
      - workdir: working directory
      - image_dir: directory to save the images
    """
    folder = os.path.join(workdir, image_dir, f"ckpt_{ckpt}")
    
    if not tf.io.gfile.exists(folder):
        tf.io.gfile.makedirs(os.path.dirname(folder))
        
    imgs = []

    for i, sample in enumerate(
        tqdm.tqdm(all_samples, desc="Saving samples", colour="green")
    ):
        # sample = np.clip(
        #     sample.permute(0, 2, 3, 1).cpu().numpy() * 255.0, 0, 255
        # ).astype(np.uint8)
        
        
      # reshape to (batch_size, image_size, image_size, num_channels)
      # type of sample is torch.Tensor
        sample = sample.view(
            (
                config.eval.batch_size,
                config.data.num_channels,
                config.data.image_size,
                config.data.image_size
            )
        )
        
        #TODO: improve this to have time on the image
        image_grid = make_grid(sample, 
                               nrow=int(np.sqrt(config.eval.batch_size)))
        
        # store every 10 images
        if i % 10 == 0:
            im = Image.fromarray(image_grid.mul_(255.)
                                 .clamp_(0, 255)
                                 .permute(1, 2, 0)
                                 .to('cpu', torch.uint8)
                                 .numpy())
            imgs.append(im)

        save_image(
            image_grid,
            os.path.join(workdir, image_dir, f"ckpt_{ckpt}", f"sample_{i}.png"),
        )
        torch.save(
            sample,
            os.path.join(workdir, image_dir, f"ckpt_{ckpt}", f"sample_raw_{i}.pt"),
        )

    # save gif
    imgs[0].save(
        os.path.join(workdir, "animation.gif"),
        save_all=True,
        append_images=imgs[1:],
        duration=1,
        loop=0,
    )

    print("GIF saved")
