from .nodes.rating_checker_gantman import RatingCheckerGantMan
from .nodes.rating_checker_marqo import RatingCheckerMarqo
from .nodes.rating_checker_nudenet import RatingCheckerNudeNet

NODE_CLASS_MAPPINGS = {
    "RatingCheckerGantMan": RatingCheckerGantMan,
    "RatingCheckerMarqo": RatingCheckerMarqo,
    "RatingCheckerNudeNet": RatingCheckerNudeNet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RatingCheckerGantMan": "Rating Checker (GantMan)",
    "RatingCheckerMarqo": "Rating Checker (Marqo)",
    "RatingCheckerNudeNet": "Rating Checker (NudeNet)",
}
