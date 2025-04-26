import numpy as np
import timm
import torch
from PIL import Image


class RatingCheckerMarqo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold_nsfw": (
                    "FLOAT",
                    {"default": 0.94, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LIST", "LIST")
    RETURN_NAMES = ("scores", "ratings")
    FUNCTION = "classify"
    CATEGORY = "utils"

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

    def classify(self, image, threshold_nsfw):
        scores = []
        ratings = []

        for img_tensor in image:
            img_np = img_tensor.permute(2, 0, 1).cpu().numpy()  # (C, H, W)
            img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C)
            img_np = (img_np * 255).astype(np.uint8)
            img = Image.fromarray(img_np)

            input_tensor = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)

            nsfw_index = self.class_names.index("NSFW")
            nsfw_score = probabilities[nsfw_index].item()

            if nsfw_score < threshold_nsfw:
                rating = "general"
            else:
                rating = "nsfw"

            scores.append(round(nsfw_score, 2))
            ratings.append(rating)

        return (scores, ratings)
