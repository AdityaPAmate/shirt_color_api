"""
detector.py

Purpose:
--------
Load the GroundingDINO model once and detect a garment in an input image.
"""

# Python built-in library
import logging
from pathlib import Path

import cv2
from api.ai.utils import log_execution_time

# GroundingDINO
from groundingdino.util.inference import Model


logger = logging.getLogger(__name__)


class ShirtDetector:
    """
    Load the GroundingDINO model once and reuse the same model instance.
    """

    # Shared model instance for all ShirtDetector objects.
    _model = None

    def __init__(self):

        # If the model is already loaded, reuse the existing model
        # instead of loading it again.
        if ShirtDetector._model is not None:
            self.model = ShirtDetector._model
            return

        # Find the project root directory so that model files can be
        # loaded correctly from the project structure.
        BASE_DIR = Path(__file__).resolve().parent.parent.parent

        # Path to the GroundingDINO model configuration file.
        config_path = (
            BASE_DIR
            / "ai_models"
            / "grounding_dino"
            / "configs"
            / "GroundingDINO_SwinT_OGC.py"
        )

        # Path to the GroundingDINO model checkpoint file.
        checkpoint_path = (
            BASE_DIR
            / "ai_models"
            / "grounding_dino"
            / "checkpoints"
            / "groundingdino_swint_ogc.pth"
        )

        # The project currently runs GroundingDINO on the CPU.
        device = "cpu"

        logger.info("Loading GroundingDINO model...")

        ShirtDetector._model = Model(
            model_config_path=str(config_path),
            model_checkpoint_path=str(checkpoint_path),
            device=device,
        )

        self.model = ShirtDetector._model

        logger.info("GroundingDINO model loaded successfully.")

    @log_execution_time
    def detect_shirt(
        self,
        image_path,
        detection_target="shirt",
        box_threshold=0.35,
        text_threshold=0.25,
    ):
        """
        Detect a garment in an image.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        detection_target : str
            Text prompt or class name given to GroundingDINO.
            Examples include "shirt", "pant", "jacket", and "kurta".
            This value can be passed from the pipeline or API.
            The default value is "shirt".

        box_threshold : float
            Minimum confidence required for object detection.

        text_threshold : float
            Minimum confidence required for matching the text prompt.

        Returns
        -------
        detections
            GroundingDINO detection result.
        """

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Unable to read image: {image_path}"
            )

        detections = self.model.predict_with_classes(
            image=image,
            classes=[detection_target],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if len(detections.xyxy) == 0:
            return None

        # Get the first detected bounding box from GroundingDINO.
        x1, y1, x2, y2 = detections.xyxy[0]

        # Read the image again to get its height and width.
        # These dimensions are used to keep the padded box inside
        # the image boundaries.
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        # Add a small padding around the detected garment so that
        # the bounding box includes a little extra area around it.
        padding = 12

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width - 1, x2 + padding)
        y2 = min(height - 1, y2 + padding)

        return {
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(detections.confidence[0]),
        }