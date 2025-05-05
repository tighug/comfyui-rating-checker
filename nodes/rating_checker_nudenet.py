import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from nudenet import NudeDetector
from PIL import ImageDraw

from ..utils.image_utils import pil_to_tensor, tensor_to_pil
from .rating_checker_gantman import RatingCheckerGantMan


class RatingCheckerNudeNet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "threshold_detect": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "detect_female_face": ("BOOLEAN", {"default": False}),
                "detect_male_face": ("BOOLEAN", {"default": False}),
                "detect_armpits": ("BOOLEAN", {"default": False}),
                "detect_female_breast": ("BOOLEAN", {"default": True}),
                "detect_male_breast": ("BOOLEAN", {"default": False}),
                "detect_belly": ("BOOLEAN", {"default": False}),
                "detect_female_genitalia": ("BOOLEAN", {"default": True}),
                "detect_male_genitalia": ("BOOLEAN", {"default": True}),
                "detect_buttocks": ("BOOLEAN", {"default": False}),
                "detect_anus": ("BOOLEAN", {"default": True}),
                "detect_feet": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "IMAGE",
    )
    RETURN_NAMES = (
        "lagels",
        "detections_json",
        "boxed_images",
    )
    FUNCTION = "detect"
    CATEGORY = "utils"
    OUTPUT_IS_LIST = (True, True, True)

    def __init__(self):
        self.model_path = Path(__file__).parents[1] / "models" / "640m.onnx"
        if self.model_path.exists():
            print("Use 640m")
            self.detector = NudeDetector(
                model_path=self.model_path, inference_resolution=640
            )
        else:
            print("Use 320m")
            self.detector = NudeDetector()
        self.gantman = RatingCheckerGantMan()

    def detect(
        self, images: torch.Tensor, threshold_detect: float, **kwargs
    ) -> tuple[str, list]:
        all_labels = []
        all_detections = []
        boxed_images = []

        # GantMan
        (labels_gant,) = self.gantman.classify(images)

        target_labels = self.build_target_labels(**kwargs)
        images_pil = tensor_to_pil(images)
        for i, image in enumerate(images_pil):
            # Detect
            with NamedTemporaryFile(suffix=".png", delete=True) as temp_file:
                image.save(temp_file.name)
                detections = self.detector.detect(temp_file.name)

            # Filter
            filtered_by_threshold = [
                d for d in detections if d["score"] >= threshold_detect
            ]
            filtered_by_target = [
                d for d in filtered_by_threshold if d["class"] in target_labels
            ]
            all_detections.append(filtered_by_target)

            # Draw bounding boxes
            draw = ImageDraw.Draw(image)
            for d in filtered_by_target:
                box = d["box"]  # [x1, y1, width, height]
                x1, y1 = int(box[0]), int(box[1])
                x2, y2 = int(x1 + box[2]), int(y1 + box[3])
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, y1 - 10), d["class"], fill="red")
            image_tensor = pil_to_tensor(image)
            boxed_images.append(image_tensor)

            # Label
            if len(filtered_by_target) != 0:
                all_labels.append("nsfw_r18")
            elif labels_gant[i] in ["hentai", "porn", "sexy"]:
                all_labels.append("nsfw_r15")
            else:
                all_labels.append("sfw")

        return (all_labels, json.dumps(all_detections, indent=2), boxed_images)

    def build_target_labels(self, **kwargs) -> list[str]:
        mapping = {
            "FACE_FEMALE": kwargs.get("detect_female_face", False),
            "FACE_MALE": kwargs.get("detect_male_face", False),
            "ARMPITS_EXPOSED": kwargs.get("detect_armpits", False),
            "FEMALE_BREAST_EXPOSED": kwargs.get("detect_female_breast", True),
            "MALE_BREAST_EXPOSED": kwargs.get("detect_male_breast", False),
            "FEMALE_GENITALIA_EXPOSED": kwargs.get("detect_female_genitalia", True),
            "MALE_GENITALIA_EXPOSED": kwargs.get("detect_male_genitalia", True),
            "BELLY_EXPOSED": kwargs.get("detect_belly", False),
            "BUTTOCKS_EXPOSED": kwargs.get("detect_buttocks", False),
            "ANUS_EXPOSED": kwargs.get("detect_anus", True),
            "FEET_EXPOSED": kwargs.get("detect_feet", False),
        }
        return [label for label, enabled in mapping.items() if enabled]
