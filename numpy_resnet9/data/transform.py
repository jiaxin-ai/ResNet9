from PIL import Image
import numpy as np


def normalize(img_path, mean, std):
    img = Image.open(img_path)
    img_np = np.array(img).astype(np.float32)

    img_np /= 255.0

    if len(img_np.shape) == 2:  # grey
        img_np = np.expand_dims(img_np, axis=0)

    img_np = (img_np - mean[0]) / std[0]

    return img_np

def denormalize_and_save(img_np, mean, std, save_path):

    img_np = img_np * std[0] + mean[0]

    img_np = img_np * 255.0

    img_np = np.squeeze(img_np, axis=0)

    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    img_pil = Image.fromarray(img_np)

    img_pil.save(save_path)
    return img_np