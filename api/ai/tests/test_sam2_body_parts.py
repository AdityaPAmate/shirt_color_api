"""
File:
    test_sam2_body_parts.py

Purpose:
    Test SAM 2.1 segmentation for:

        - Torso
        - Left Hand
        - Right Hand

    Uses positive and negative point prompts so that
    SAM does not return the complete person silhouette.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ----------------------------------------------------------
# Input / Output paths
# ----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

PERSON_IMAGE = (
    BASE_DIR
    / "test_images"
    / "person3.jpg"
)

MASKS_OUTPUT_DIR = (
    BASE_DIR
    / "test_images"
    / "sam2_body_part_masks"
)

MASKS_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ----------------------------------------------------------
# SAM 2.1 paths
# ----------------------------------------------------------

SAM2_CONFIG = (
    BASE_DIR
    / "ai_models"
    / "sam2"
    / "configs"
    / "sam2.1_hiera_t.yaml"
)

SAM2_CHECKPOINT = (
    BASE_DIR
    / "ai_models"
    / "sam2"
    / "checkpoints"
    / "sam2.1_hiera_tiny.pt"
)


# ----------------------------------------------------------
# Device
# ----------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------
# Save mask
# ----------------------------------------------------------

def save_mask(mask, output_path):

    mask_image = (
        mask.astype(np.uint8) * 255
    )

    cv2.imwrite(
        str(output_path),
        mask_image
    )


# ----------------------------------------------------------
# Print mask information
# ----------------------------------------------------------

def print_mask_information(mask, label):

    ys, xs = np.where(mask)

    if len(xs) == 0:

        print(f"\n{label}: EMPTY MASK")
        return

    x1 = int(xs.min())
    y1 = int(ys.min())

    x2 = int(xs.max())
    y2 = int(ys.max())

    width = x2 - x1 + 1
    height = y2 - y1 + 1

    area = int(mask.sum())

    print("\nMask Information")
    print("-----------------------------")
    print(f"Label       : {label}")
    print(f"X1          : {x1}")
    print(f"Y1          : {y1}")
    print(f"X2          : {x2}")
    print(f"Y2          : {y2}")
    print(f"Width       : {width}")
    print(f"Height      : {height}")
    print(f"Area        : {area}")


# ----------------------------------------------------------
# Run SAM 2.1
# ----------------------------------------------------------

def run_sam2_body_parts(
    person_image_path,
    output_dir
):

    # ------------------------------------------------------
    # 1. Check files
    # ------------------------------------------------------

    if not Path(person_image_path).exists():
        raise FileNotFoundError(
            f"Person image not found:\n"
            f"{person_image_path}"
        )

    if not SAM2_CONFIG.exists():
        raise FileNotFoundError(
            f"SAM config not found:\n"
            f"{SAM2_CONFIG}"
        )

    if not SAM2_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found:\n"
            f"{SAM2_CHECKPOINT}"
        )

    # ------------------------------------------------------
    # 2. Load SAM 2.1
    # ------------------------------------------------------

    print("\nLoading SAM 2.1...")

    sam_model = build_sam2(
        config_file=str(SAM2_CONFIG),
        ckpt_path=str(SAM2_CHECKPOINT),
        device=DEVICE,
    )

    predictor = SAM2ImagePredictor(
        sam_model
    )

    print("SAM 2.1 Loaded Successfully.")

    # ------------------------------------------------------
    # 3. Read image
    # ------------------------------------------------------

    image = cv2.imread(
        str(person_image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Unable to read image:\n"
            f"{person_image_path}"
        )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    height, width = image_rgb.shape[:2]

    print("\nImage Information")
    print("-----------------------------")
    print(f"Width  : {width}")
    print(f"Height : {height}")

    # ------------------------------------------------------
    # 4. Give image to SAM
    # ------------------------------------------------------

    predictor.set_image(
        image_rgb
    )

    # ======================================================
    # IMPORTANT
    # ======================================================
    #
    # These coordinates are EXAMPLES.
    #
    # You must adjust them according to your person image.
    #
    # point format:
    #
    # [x, y]
    #
    # label:
    #
    # 1 = positive / foreground
    # 0 = negative / background
    #
    # ======================================================

    body_parts = {

        # --------------------------------------------------
        # TORSO
        # --------------------------------------------------

        "Torso": {

            "points": np.array([
                # Positive point - chest
                [
                    width // 2,
                    int(height * 0.43)
                ],

                # Negative point - head
                [
                    width // 2,
                    int(height * 0.12)
                ],

                # Negative point - left hand/arm
                [
                    int(width * 0.18),
                    int(height * 0.60)
                ],

                # Negative point - right hand/arm
                [
                    int(width * 0.82),
                    int(height * 0.60)
                ],
            ], dtype=np.float32),

            "labels": np.array([
                1,
                0,
                0,
                0
            ], dtype=np.int32)
        },


        # --------------------------------------------------
        # LEFT HAND
        # --------------------------------------------------

        "Left_Hand": {

            "points": np.array([

                # Positive point - left hand
                [
                    int(width * 0.18),
                    int(height * 0.70)
                ],

                # Negative - torso
                [
                    width // 2,
                    int(height * 0.45)
                ],

                # Negative - head
                [
                    width // 2,
                    int(height * 0.12)
                ],

                # Negative - right side
                [
                    int(width * 0.82),
                    int(height * 0.70)
                ],

            ], dtype=np.float32),

            "labels": np.array([
                1,
                0,
                0,
                0
            ], dtype=np.int32)
        },


        # --------------------------------------------------
        # RIGHT HAND
        # --------------------------------------------------

        "Right_Hand": {

            "points": np.array([

                # Positive point - right hand
                [
                    int(width * 0.82),
                    int(height * 0.70)
                ],

                # Negative - torso
                [
                    width // 2,
                    int(height * 0.45)
                ],

                # Negative - head
                [
                    width // 2,
                    int(height * 0.12)
                ],

                # Negative - left side
                [
                    int(width * 0.18),
                    int(height * 0.70)
                ],

            ], dtype=np.float32),

            "labels": np.array([
                1,
                0,
                0,
                0
            ], dtype=np.int32)
        }
    }

    # ------------------------------------------------------
    # 5. Process each body part
    # ------------------------------------------------------

    saved_paths = {}

    for label, prompt in body_parts.items():

        print("\n==========================================")
        print(f"Processing: {label}")
        print("==========================================")

        point_coords = prompt["points"]
        point_labels = prompt["labels"]

        print("\nPrompt Points:")

        for point, point_label in zip(
            point_coords,
            point_labels
        ):

            point_type = (
                "POSITIVE"
                if point_label == 1
                else "NEGATIVE"
            )

            print(
                f"{point_type:10s} "
                f"-> ({int(point[0])}, {int(point[1])})"
            )

        # --------------------------------------------------
        # SAM prediction
        # --------------------------------------------------

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # --------------------------------------------------
        # Best mask
        # --------------------------------------------------

        best_index = int(
            np.argmax(scores)
        )

        best_mask = masks[
            best_index
        ]

        best_score = float(
            scores[best_index]
        )

        print(
            f"\nSAM Score: {best_score:.4f}"
        )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        output_path = (
            Path(output_dir)
            / f"{label}.png"
        )

        save_mask(
            best_mask,
            output_path
        )

        saved_paths[label] = str(
            output_path
        )

        print(
            f"Saved: {output_path}"
        )

        # --------------------------------------------------
        # Mask information
        # --------------------------------------------------

        print_mask_information(
            best_mask,
            label
        )

    # ------------------------------------------------------
    # Final
    # ------------------------------------------------------

    print("\n==========================================")
    print("ALL MASKS GENERATED")
    print("==========================================")

    for label, path in saved_paths.items():

        print(
            f"{label:15s} -> {path}"
        )

    return saved_paths


# ----------------------------------------------------------
# Run
# ----------------------------------------------------------

if __name__ == "__main__":

    run_sam2_body_parts(
        PERSON_IMAGE,
        MASKS_OUTPUT_DIR
    )