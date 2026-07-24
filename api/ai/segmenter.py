"""
segmenter.py

Purpose:
--------
Load the SAM 2.1 model once for image segmentation.
"""

from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class ShirtSegmenter:
    """
    Loads the SAM 2.1 model once and reuses it.
    """

    _predictor = None

    def __init__(self):

        if ShirtSegmenter._predictor is not None:
            self.predictor = ShirtSegmenter._predictor
            return

        # Project root directory
        BASE_DIR = Path(__file__).resolve().parent.parent.parent

        config_path = (
            BASE_DIR
            / "ai_models"
            / "sam2"
            / "configs"
            / "sam2.1_hiera_t.yaml"
        )

        checkpoint_path = (
            BASE_DIR
            / "ai_models"
            / "sam2"
            / "checkpoints"
            / "sam2.1_hiera_tiny.pt"
        )

        device = "cpu"

        print("Loading SAM 2.1...")

        sam_model = build_sam2(
            config_file=str(config_path),
            ckpt_path=str(checkpoint_path),
            device=device,
        )

        ShirtSegmenter._predictor = SAM2ImagePredictor(sam_model)
        self.predictor = ShirtSegmenter._predictor

        print("SAM 2.1 Loaded Successfully.")