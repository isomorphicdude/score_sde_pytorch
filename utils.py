import torch
import tensorflow as tf
import os
import logging
import tqdm
import numpy as np
from torchvision.utils import save_image, make_grid
import PIL


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


def save_gif(all_samples, config, ckpt, image_dir):
    """Save samples as a GIF file."""
    imgs = []

    for i, sample in enumerate(
        tqdm.tqdm(all_samples, desc="Saving samples", colour="green")
    ):
        sample = np.clip(
            sample.permute(0, 2, 3, 1).cpu().numpy() * 255.0, 0, 255
        ).astype(np.uint8)

        sample = sample.reshape(
            (
                -1,
                config.data.image_size,
                config.data.image_size,
                config.data.num_channels,
            )
        )

        image_grid = make_grid(sample, nrow=torch.sqrt(config.eval.batch_size).int())
        # store every 10 images
        if i % 10 == 0:
            im = PIL.Image.fromarrray(image_grid)
            imgs.append(im)

        save_image(
            image_grid,
            os.path.join(image_dir, f"ckpt_{ckpt}", f"sample_{i}.png"),
        )
        torch.save(
            sample,
            os.path.join(image_dir, f"ckpt_{ckpt}", f"sample_raw_{i}.pt"),
        )

    # save gif
    imgs[0].save(
        "animation.gif",
        save_all=True,
        append_images=imgs[1:],
        duration=1,
        loop=0,
    )

    print("GIF saved")
