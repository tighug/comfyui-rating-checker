from pathlib import Path

import numpy as np
import onnxruntime
import requests
import torch
from loguru import logger
from PIL import Image

from ..utils.image_utils import tensor_to_pil

MODEL_URL = (
    "https://github.com/iola1999/nsfw-detect-onnx/releases/download/v1.0.0/model.onnx"
)


class RatingCheckerGantMan:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("labels",)
    FUNCTION = "classify"
    CATEGORY = "utils"
    OUTPUT_IS_LIST = (True,)

    def __init__(self):
        self.model_path = Path(__file__).parents[1] / "models" / "nsfw_detect.onnx"
        self.ensure_model_downloaded()
        self.session = onnxruntime.InferenceSession(str(self.model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.classes = ["drawings", "hentai", "neutral", "porn", "sexy"]

    def classify(self, images: torch.Tensor) -> tuple[list[str]]:
        labels = []

        images_pil = tensor_to_pil(images)
        for image in images_pil:
            input = self.preprocess(image)
            output = self.session.run(None, {self.input_name: input})[0]
            index = int(np.argmax(output[0]))
            label = self.classes[index]
            labels.append(label)

        return (labels,)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.resize((299, 299)).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # normalize to [-1, 1]
        img = np.expand_dims(img, axis=0)  # NHWC (1, 299, 299, 3)
        return img.astype(np.float32)

    def ensure_model_downloaded(self) -> None:
        if self.model_path.exists():
            return

        logger.info("[RatingChecker] nsfw_detect.onnx not found, downloading...")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with requests.get(MODEL_URL, stream=True) as response:
                response.raise_for_status()
                with open(self.model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            logger.info(f"[RatingChecker] Downloaded model to {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")
