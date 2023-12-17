from PIL import Image
import numpy as np
import torch

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image


if __name__ == "__main__":

    img_path = "./images/zeyu.png"
    landmark_save_path = "./landmarks/zeyu_lm.png"
    output_save_path = "./outputs/sample_2.png"
    # Stable Diffusion 2.1-base:
    controlnet = ControlNetModel.from_pretrained("CrucibleAI/ControlNetMediaPipeFace", torch_dtype=torch.float16, variant="fp16")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
    )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(torch.device("cuda"))

    face_landmark = load_image(landmark_save_path)

    image = pipe("A Hulk", image=face_landmark, num_inference_steps=30).images[0]
    image.save(output_save_path)
    