from PIL import Image
import numpy as np
import cv2

def load_image(path):
    return Image.open(path).convert("RGB")

def save_image(img, path):
    img.save(path)

def generate_canny(img):
    img_cv = np.array(img)
    edges = cv2.Canny(img_cv, 100, 200)
    return Image.fromarray(edges)
