import json

import numpy as np
import timm
import torch
from nudenet import NudeDetector
from torchvision import transforms


class RatingCheckerNudeNet:
    def __init__(self):
        self.detector = NudeDetector()
        self.model = timm.create_model(
            "hf_hub:Marqo/nsfw-image-detection-384", pretrained=True
        )
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "threshold_nsfw": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "threshold_detect": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "detect_armpits": ("BOOLEAN", {"default": False}),
                "detect_female_breast": ("BOOLEAN", {"default": True}),
                "detect_male_breast": ("BOOLEAN", {"default": False}),
                "detect_female_genitalia": ("BOOLEAN", {"default": True}),
                "detect_male_genitalia": ("BOOLEAN", {"default": True}),
                "detect_belly": ("BOOLEAN", {"default": False}),
                "detect_buttocks": ("BOOLEAN", {"default": False}),
                "detect_anus": ("BOOLEAN", {"default": True}),
                "detect_feet": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "detections_json",
        "nsfw_labels",
    )
    FUNCTION = "detect"
    CATEGORY = "utils"

    def detect(
        self,
        images,
        detect_min_score,
        nsfw_threshold,
        detect_armpits,
        detect_female_breast,
        detect_male_breast,
        detect_female_genitalia,
        detect_male_genitalia,
        detect_belly,
        detect_buttocks,
        detect_anus,
        detect_feet,
    ):
        all_detections = []
        nsfw_labels = []

        allowed_classes = []
        if detect_armpits:  # 脇
            allowed_classes.append("ARMPITS_EXPOSED")
        if detect_female_breast:  # 女性の胸
            allowed_classes.append("FEMALE_BREAST_EXPOSED")
        if detect_male_breast:  # 男性の胸
            allowed_classes.append("MALE_BREAST_EXPOSED")
        if detect_female_genitalia:  # 女性器
            allowed_classes.append("FEMALE_GENITALIA_EXPOSED")
        if detect_male_genitalia:  # 男性器
            allowed_classes.append("MALE_GENITALIA_EXPOSED")
        if detect_belly:  # お腹
            allowed_classes.append("BELLY_EXPOSED")
        if detect_buttocks:  # お尻
            allowed_classes.append("BUTTOCKS_EXPOSED")
        if detect_anus:  # 肛門
            allowed_classes.append("ANUS_EXPOSED")
        if detect_feet:  # 足
            allowed_classes.append("FEET_EXPOSED")

        for image in images:
            np_image = (image.cpu().numpy() * 255).astype(np.uint8)
            detections = self.detector.detect(np_image)
            filtered = [d for d in detections if d["score"] >= detect_min_score]
            final_detections = [
                {"class": d["class"], "score": round(d["score"], 2)}
                for d in filtered
                if d["class"] in allowed_classes
            ]

            input_tensor = self.transform(np_image).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=-1)
                score = probs.max().item()

            all_detections.append(
                {"nudenet_detections": final_detections, "nsfw_score": round(score, 2)}
            )

            if len(final_detections) > 0:
                nsfw_labels.append("nsfw(R-18)")
            elif score >= nsfw_threshold:
                nsfw_labels.append("nsfw(R-15)")
            else:
                nsfw_labels.append("sfw")

        return (json.dumps(all_detections, indent=2), json.dumps(nsfw_labels))
