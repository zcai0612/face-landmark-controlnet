from typing import Union
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from utils.face_landmark import LandmarkAnnotator

if __name__ == "__main__":
    img_path = "./images/zeyu.png"
    landmark_save_path = "./landmarks/zeyu_lm.png"

    anno = LandmarkAnnotator()

    anno.get_annotation(
        image=img_path,
        resolution=(768, 768),
        save_path=landmark_save_path
    )