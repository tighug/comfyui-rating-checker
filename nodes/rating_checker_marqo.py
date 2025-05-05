import timm
import torch

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
        "STRING",
        "FLOAT",
    )
    RETURN_NAMES = (
        "labels",
        "scores",
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
        ).eval()
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def classify(
        self, images: torch.Tensor, threshold_nsfw: float
    ) -> tuple[list[str], list[float]]:
        labels = []
        scores = []

        images_pil = tensor_to_pil(images)
        for image in images_pil:
            with torch.no_grad():
                image_transformed = self.transforms(image).unsqueeze(0)
                output = self.model(image_transformed).softmax(dim=-1).cpu()
                label_names = self.model.pretrained_cfg["label_names"]
                index_nsfw = label_names.index("NSFW")
                score_nsfw = round(output[0][index_nsfw].item(), 3)
                label = "sfw" if score_nsfw < threshold_nsfw else "nsfw"

            labels.append(label)
            scores.append(score_nsfw)

        return (labels, scores)
