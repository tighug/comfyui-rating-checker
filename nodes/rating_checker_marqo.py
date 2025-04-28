from typing import List, Tuple

import timm
import torch
from PIL import Image

from ..utils.image_utils import tensor_to_pil


class RatingCheckerMarqo:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "images": ("IMAGE",),
                "threshold_nsfw": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = (
        "FLOAT",
        "STRING",
    )
    RETURN_NAMES = (
        "scores",
        "ratings",
    )
    FUNCTION = "classify"
    CATEGORY = "utils"
    OUTPUT_IS_LIST = (
        True,
        True,
    )

    def __init__(self):
        self.model = timm.create_model(
            "hf_hub:Marqo/nsfw-image-detection-384", pretrained=True
        )
        self.model.eval()
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(
            **self.data_config, is_training=False
        )
        self.class_names = self.model.pretrained_cfg["label_names"]
        self.nsfw_index = self.class_names.index("NSFW")

    def classify(
        self, images: torch.Tensor, threshold_nsfw: float
    ) -> Tuple[List[float], List[str]]:
        scores = []
        ratings = []

        for img_tensor in images:
            score, rating = self._classify_single(img_tensor, threshold_nsfw)
            scores.append(score)
            ratings.append(rating)

        return scores, ratings

    def _classify_single(
        self, img_tensor: torch.Tensor, threshold_nsfw: float
    ) -> Tuple[float, str]:
        img = tensor_to_pil(img_tensor)
        nsfw_score = self._predict_nsfw_score(img)
        rating = "general" if nsfw_score < threshold_nsfw else "nsfw"
        return round(nsfw_score, 2), rating

    def _predict_nsfw_score(self, img: Image.Image) -> float:
        input_tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        nsfw_score = probabilities[self.nsfw_index].item()
        return nsfw_score
