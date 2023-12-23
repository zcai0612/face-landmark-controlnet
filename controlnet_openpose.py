from PIL import Image
import numpy as np
import torch

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image


if __name__ == "__main__":

    img_path = "./images/sd_test4.png"
    #landmark_save_path = "./landmarks/sd_test_lm.png"
    output_save_path = "./outputs/sample_sd.png"
    # Stable Diffusion 2.1-base:
    controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-openposev2-diffusers")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet
    )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(torch.device("cuda"))

    face_landmark = load_image(img_path)

    image = pipe("A Batman", image=face_landmark, num_inference_steps=30, generator=torch.manual_seed(42)).images[0]
    image.save(output_save_path)
    