from annotator import generate_annotation
from PIL import Image

if __name__ == "__main__":
    face_img = Image.open("./images/zeyu.png").convert("RGB")
    face_lm = generate_annotation(face_img, max_faces=1)
    Image.fromarray(face_lm).save("./landmarks/zeyu.png")