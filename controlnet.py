from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

image = load_image(
    "https://huggingface.co/CrucibleAI/ControlNetMediaPipeFace/resolve/main/samples_laion_face_dataset/family_annotation.png"
)

# Stable Diffusion 2.1-base:
controlnet = ControlNetModel.from_pretrained("CrucibleAI/ControlNetMediaPipeFace", torch_dtype=torch.float16, variant="fp16")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)