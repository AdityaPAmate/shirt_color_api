"""
segmenter.py

Purpose:
--------
Load the SAM 2.1 model once for image segmentation.
"""

from pathlib import Path

import cv2
import numpy as np

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

    def segment_shirt(self, image_path, box):
        """
        Generate a segmentation mask for the shirt.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        box : list
            Bounding box from GroundingDINO.
            Format: [x1, y1, x2, y2]

        Returns
        -------
        numpy.ndarray
            Binary mask of the shirt.
        """

        # Read image
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        # Convert BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Give image to SAM
        self.predictor.set_image(image)

        # Convert box to NumPy array
        input_box = np.array(box)

        # Predict mask
        masks, scores, logits = self.predictor.predict(
            box=input_box,
            multimask_output=True,
        )

        # Select the mask with the highest confidence score
        best_index = np.argmax(scores)

        return masks[best_index]


    def save_mask(self, mask, output_path):
        """
        Save the binary mask as an image.
        """

        import cv2
        import numpy as np

        mask_image = (mask * 255).astype(np.uint8)

        cv2.imwrite(output_path, mask_image)