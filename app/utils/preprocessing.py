import torch
import numpy as np
from PIL import Image

IMG_SIZE = 224

def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor  # shape: [1, 3, 224, 224]
