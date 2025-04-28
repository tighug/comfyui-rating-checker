import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    img = tensor.permute(2, 0, 1).cpu().numpy()  # (C, H, W)
    img = np.transpose(img, (1, 2, 0))  # (H, W, C)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)
