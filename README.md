# ComfyUI Rating Checker

[[日本語]](.//README_JP.md)

A custom node for ComfyUI that classifies images into NSFW (Not Safe For Work) categories.

## Features

This node is specifically designed to classify illustration-style images for NSFW content.

Existing similar nodes offer high accuracy for real human photographs but tend to overclassify anime-style illustrations as NSFW, even with minimal nudity. Moreover, fine-grained categorization (such as R15 / R18) has been challenging.

To address these issues, `Rating Checker (NudeNet)` combines an object detection model (NudeNet) with a scoring model (Marqo) to classify images into three labels: "SFW / NSFW (R15) / NSFW (R18)."

## Installation

### ComfyUI Manager

Not supported.

### Manual

Clone the repository into your `custom_nodes` directory:

```bash
git clone https://github.com/tighug/comfyui-eagle-feeder.git
```

## Usage

Three nodes are included for NSFW rating. Primarily, the NudeNet version is intended for use, but the other two nodes created during verification are also bundled.

### Rating Checker (NudeNet)

Classifies images into the following three labels based on specific conditions:

- `nsfw_r18`: Detects any of the following body parts:
  - armpits
  - female_breast
  - male_breast
  - female_genitalia
  - male_genitalia
  - belly
  - buttocks
  - anus
  - feet
- `nsfw_r15`: If not `nsfw_r18`, but `nsfw_score > threshold_nsfw`
- `sfw`: If none of the above conditions are met

![NudeNet R15](./doc/images/nudenet.png)

![NudeNet R18](./doc/images/nudenet_r18.png)

Models used:

- [notAI-tech/NudeNet](https://github.com/notAI-tech/NudeNet/tree/v3)
- [Marqo/nsfw-image-detection-384](https://huggingface.co/Marqo/nsfw-image-detection-384)

### Rating Checker (GantMan)

Classifies images into the following five labels:

- `drawings`: Illustrations
- `hentai`: Anime or manga
- `neutral`: General images
- `porn`: Realistic sexual images
- `sexy`: Images with sexual undertones

![GantMan](./doc/images/gantman.png)

**Note:**
Anime-style illustrations are classified as either `drawings` or `hentai`, but often end up categorized as `hentai` even without swimsuits or lingerie.
Thus, this model is not well-suited for illustration-specific classification, but it is effective for distinguishing between real photos and illustrations and for NSFW detection in real images.

Model used:

- [GantMan/nsfw_model](https://github.com/GantMan/nsfw_model)

### Rating Checker (Marqo)

Calculates an NSFW score for the image and outputs it as `scores`.
It also performs binary classification (`sfw` / `nsfw`) based on the `threshold_nsfw` value and outputs it as `ratings`.

![Marqo](./doc/images/marqo.png)

Model used:

- [Marqo/nsfw-image-detection-384](https://huggingface.co/Marqo/nsfw-image-detection-384)

## License

[MIT](./LICENSE)
