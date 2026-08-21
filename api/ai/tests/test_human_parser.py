"""
File:
    test_human_parser.py  (exploratory/test script — not yet part of the pipeline)

Purpose:
    Run fashn-ai/fashn-human-parser on a sample person image and save each
    predicted semantic-class mask as a separate PNG, so we can visually
    inspect which label (Torso, Left-arm, Right-arm, Upper-clothes, etc.)
    corresponds to which region of the body.
"""

from pathlib import Path
from transformers import pipeline
from PIL import Image

# ----------------------------------------------------------
# Input / Output paths
# ----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]

PERSON_IMAGE = f"{BASE_DIR}/test_images/ladies1.jpg"

# TODO: तुम्ही सांगितलेली output location इथे टाका
MASKS_OUTPUT_DIR = f"{BASE_DIR}/test_images/human_parser_masks"

Path(MASKS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def run_human_parser(person_image_path: str, output_dir: str):
    """
    person_image_path: input person photo चा path
    output_dir: प्रत्येक class ची mask image कुठे save करायची

    Return करतो: dict {label_name: saved_mask_path}
    """
    # ------------------------------------------------------
    # 1. Model load करणं (पहिल्यांदा चालवताना download होईल,
    #    नंतर local cache मधून लोड होईल — इंटरनेट लागणार नाही)
    # ------------------------------------------------------
    parser_pipe = pipeline(
        task="image-segmentation",
        model="fashn-ai/fashn-human-parser",
    )

    # ------------------------------------------------------
    # 2. Person image वर inference चालवणं
    # ------------------------------------------------------
    image = Image.open(person_image_path).convert("RGB")
    results = parser_pipe(image)

    # results = list of dicts:
    #   [{"label": "Torso", "mask": <PIL Image, mode 'L'>, "score": 0.98}, ...]

    saved_paths = {}

    # ------------------------------------------------------
    # 3. प्रत्येक class ची mask वेगळी PNG म्हणून save करणं
    # ------------------------------------------------------
    for item in results:
        label = item["label"].replace(" ", "_")
        mask_img = item["mask"]  # PIL Image, single-channel (0/255)

        save_path = f"{output_dir}/{label}.png"
        mask_img.save(save_path)

        saved_paths[label] = save_path
        print(f"Saved: {label:20s} -> {save_path}")

    return saved_paths


if __name__ == "__main__":
    run_human_parser(PERSON_IMAGE, MASKS_OUTPUT_DIR)