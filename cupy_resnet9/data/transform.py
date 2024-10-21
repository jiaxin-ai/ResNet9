from PIL import Image
import cupy as cp


def normalize(img_path, mean, std):
    img = Image.open(img_path)
    img_np = cp.array(img).astype(cp.float32)

    img_np /= 255.0

    if len(img_np.shape) == 2:  # grey
        img_np = cp.expand_dims(img_np, axis=0)

    img_np = (img_np - mean[0]) / std[0]

    return img_np

def denormalize_and_save(img_np, mean, std, save_path):

    img_np = img_np * std[0] + mean[0]

    img_np = img_np * 255.0

    img_np = cp.squeeze(img_np, axis=0)

    img_np = cp.clip(img_np, 0, 255).astype(cp.uint8)

    img_pil = Image.fromarray(img_np)

    img_pil.save(save_path)
    return img_np