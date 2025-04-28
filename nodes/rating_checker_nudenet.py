import json
from typing import Dict, List, Tuple

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

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "nsfw_labels",
        "detections_json",
    )
    FUNCTION = "detect"
    CATEGORY = "utils"
    OUTPUT_IS_LIST = (
        True,
        True,
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "images": ("IMAGE",),
                "threshold_nsfw": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "threshold_detect": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},
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

    def detect(
        self,
        images: torch.Tensor,
        threshold_nsfw: float,
        threshold_detect: float,
        detect_armpits: bool,
        detect_female_breast: bool,
        detect_male_breast: bool,
        detect_female_genitalia: bool,
        detect_male_genitalia: bool,
        detect_belly: bool,
        detect_buttocks: bool,
        detect_anus: bool,
        detect_feet: bool,
    ) -> Tuple[str, str]:
        allowed_classes = self._build_allowed_classes(
            detect_armpits,
            detect_female_breast,
            detect_male_breast,
            detect_female_genitalia,
            detect_male_genitalia,
            detect_belly,
            detect_buttocks,
            detect_anus,
            detect_feet,
        )

        all_detections = []
        nsfw_labels = []

        for image in images:
            nsfw_label, detection = self._process_single_image(
                image, allowed_classes, threshold_detect, threshold_nsfw
            )
            all_detections.append(detection)
            nsfw_labels.append(nsfw_label)

        return nsfw_labels, json.dumps(all_detections, indent=2)

    def _build_allowed_classes(
        self,
        detect_armpits: bool,
        detect_female_breast: bool,
        detect_male_breast: bool,
        detect_female_genitalia: bool,
        detect_male_genitalia: bool,
        detect_belly: bool,
        detect_buttocks: bool,
        detect_anus: bool,
        detect_feet: bool,
    ) -> List[str]:
        mapping = {
            "ARMPITS_EXPOSED": detect_armpits,
            "FEMALE_BREAST_EXPOSED": detect_female_breast,
            "MALE_BREAST_EXPOSED": detect_male_breast,
            "FEMALE_GENITALIA_EXPOSED": detect_female_genitalia,
            "MALE_GENITALIA_EXPOSED": detect_male_genitalia,
            "BELLY_EXPOSED": detect_belly,
            "BUTTOCKS_EXPOSED": detect_buttocks,
            "ANUS_EXPOSED": detect_anus,
            "FEET_EXPOSED": detect_feet,
        }
        return [cls for cls, enabled in mapping.items() if enabled]

    def _process_single_image(
        self,
        image: torch.Tensor,
        allowed_classes: List[str],
        threshold_detect: float,
        threshold_nsfw: float,
    ) -> Tuple[str, Dict]:
        np_image = (image.cpu().numpy() * 255).astype(np.uint8)
        detections = self.detector.detect(np_image)
        filtered = [d for d in detections if d["score"] >= threshold_detect]
        final_detections = [
            {"class": d["class"], "score": round(d["score"], 2)}
            for d in filtered
            if d["class"] in allowed_classes
        ]

        input_tensor = self.transform(np_image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=-1)
            nsfw_score = probs.max().item()

        if final_detections:
            nsfw_label = "nsfw_r18"
        elif nsfw_score >= threshold_nsfw:
            nsfw_label = "nsfw_r15"
        else:
            nsfw_label = "sfw"

        return nsfw_label, {
            "detections": final_detections,
            "nsfw_score": round(nsfw_score, 2),
        }
