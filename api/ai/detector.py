"""
detector.py

Purpose:
--------
Load the GroundingDINO model once and detect a shirt in an input image.
"""

# Python built-in library
from pathlib import Path

import cv2

# GroundingDINO
from groundingdino.util.inference import Model




class ShirtDetector:
    """
    Loads the GroundingDINO model only once and reuses it.
    """

    # Shared model instance for all ShirtDetector objects
    _model = None

    def __init__(self):

        # If the model is already loaded, reuse it
        if ShirtDetector._model is not None:
            self.model = ShirtDetector._model
            return

        # Project root directory
        BASE_DIR = Path(__file__).resolve().parent.parent.parent

        # Configuration file
        config_path = (
            BASE_DIR
            / "ai_models"
            / "grounding_dino"
            / "configs"
            / "GroundingDINO_SwinT_OGC.py"
        )

        # Model checkpoint
        checkpoint_path = (
            BASE_DIR
            / "ai_models"
            / "grounding_dino"
            / "checkpoints"
            / "groundingdino_swint_ogc.pth"
        )

        # Device
        device = "cpu"

        print("Loading GroundingDINO...")

        ShirtDetector._model = Model(
            model_config_path=str(config_path),
            model_checkpoint_path=str(checkpoint_path),
            device=device,
        )

        self.model = ShirtDetector._model

        print("GroundingDINO Loaded Successfully.")

    def detect_shirt(
            self,
            image_path,
            box_threshold=0.35,
            text_threshold=0.25,
    ):
        """
        Detect shirts in an image.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        box_threshold : float
            Minimum confidence required for object detection.

        text_threshold : float
            Minimum confidence required for text matching.

        Returns
        -------
        detections
            GroundingDINO detection result.
        """

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        detections = self.model.predict_with_classes(
            image=image,
            classes=["shirt"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if len(detections.xyxy) == 0:
            return None

        return {
            "box": detections.xyxy[0].tolist(),
            "confidence": float(detections.confidence[0]),
        }

