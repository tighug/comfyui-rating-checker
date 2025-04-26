import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image


class RatingCheckerGantMan:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("ratings",)
    FUNCTION = "classify"
    CATEGORY = "utils"

    def __init__(self):
        self.model_path = Path(__file__).parent / "models" / "nsfw_detect.onnx"
        self._ensure_model_downloaded()
        self.session = onnxruntime.InferenceSession(str(self.model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.classes = ["drawings", "hentai", "neutral", "porn", "sexy"]

    def classify(self, image):
        ratings = []
        for img_tensor in image:
            img_tensor = img_tensor.permute(2, 0, 1).cpu().numpy()  # (C, H, W)
            img_tensor = np.transpose(img_tensor, (1, 2, 0))  # (H, W, C)
            img_tensor = (img_tensor * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_tensor)
            input_tensor = self._preprocess(img_pil)

            output = self.session.run(None, {self.input_name: input_tensor})[0]
            idx = int(np.argmax(output[0]))
            rating = self.classes[idx]

            ratings.append(rating)

        return (ratings,)

    def _ensure_model_downloaded(self):
        if not self.model_path.exists():
            print("[RatingChecker] nsfw_detect.onnx not found, downloading...")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://github.com/iola1999/nsfw-detect-onnx/releases/download/v1.0.0/model.onnx"
            try:
                urllib.request.urlretrieve(url, str(self.model_path))
                print(f"[RatingChecker] Downloaded model to {self.model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")

    def _preprocess(self, image: Image.Image):
        img = image.resize((299, 299)).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # normalize to [-1, 1]
        img = np.expand_dims(img, axis=0)  # NHWC (1, 299, 299, 3)
        return img.astype(np.float32)
