import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime
import torch
from PIL import Image

from ..utils.image_utils import tensor_to_pil


class RatingCheckerGantMan:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ratings",)
    FUNCTION = "classify"
    CATEGORY = "utils"
    OUTPUT_IS_LIST = (True,)

    def __init__(self):
        self.model_path = Path(__file__).parent / "models" / "nsfw_detect.onnx"
        self._ensure_model_downloaded()
        model_path_str = str(self.model_path)
        self.session = onnxruntime.InferenceSession(model_path_str)
        self.input_name = self.session.get_inputs()[0].name
        self.classes = ["drawings", "hentai", "neutral", "porn", "sexy"]

    def classify(self, images: torch.Tensor) -> Tuple[List[str]]:
        return ([self._classify_single(img_tensor) for img_tensor in images],)

    def _classify_single(self, img_tensor: torch.Tensor) -> str:
        img_pil = tensor_to_pil(img_tensor)
        input_tensor = self._preprocess(img_pil)
        output = self.session.run(None, {self.input_name: input_tensor})[0]
        idx = int(np.argmax(output[0]))
        return self.classes[idx]

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.resize((299, 299)).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # normalize to [-1, 1]
        img = np.expand_dims(img, axis=0)  # NHWC (1, 299, 299, 3)
        return img.astype(np.float32)

    def _ensure_model_downloaded(self) -> None:
        if not self.model_path.exists():
            print("[RatingChecker] nsfw_detect.onnx not found, downloading...")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://github.com/iola1999/nsfw-detect-onnx/releases/download/v1.0.0/model.onnx"
            try:
                urllib.request.urlretrieve(url, str(self.model_path))
                print(f"[RatingChecker] Downloaded model to {self.model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")
