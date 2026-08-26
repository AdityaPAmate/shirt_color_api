"""
segmenter.py

Purpose:
--------
Load the SAM 2.1 model once and use it for image segmentation.
"""

import logging
from pathlib import Path
import torch

import cv2
import numpy as np

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from api.ai.utils import log_execution_time

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


logger = logging.getLogger(__name__)


class ShirtSegmenter:
    """
    Load the SAM 2.1 model once and reuse the same predictor.
    """

    _predictor = None

    def __init__(self):

        # Reuse the existing predictor if the model has already been loaded.
        if ShirtSegmenter._predictor is not None:
            self.predictor = ShirtSegmenter._predictor
            return

        # Find the project root directory so the model files can be
        # loaded from the expected project locations.
        BASE_DIR = Path(__file__).resolve().parent.parent.parent

        # Path to the SAM 2.1 model configuration file.
        config_path = (
            BASE_DIR
            / "ai_models"
            / "sam2"
            / "configs"
            / "sam2.1_hiera_t.yaml"
        )

        # Path to the SAM 2.1 model checkpoint file.
        checkpoint_path = (
            BASE_DIR
            / "ai_models"
            / "sam2"
            / "checkpoints"
            / "sam2.1_hiera_tiny.pt"
        )

        # The project currently runs SAM 2.1 on the CPU.
        # The project currently runs SAM 2.1 on the CPU.
        device = "cpu"

        logger.info("Loading SAM 2.1 model...")

        # Hydra's compose() (called inside build_sam2) cannot resolve an
        # absolute filesystem path as config_file -- this is a known
        # Hydra/SAM2 limitation (it strips the leading "/" and looks for
        # the rest relative to its own registered search paths, so it
        # never finds a real absolute path). The fix: register the
        # config's actual directory as a Hydra search path ourselves,
        # then pass just the filename to build_sam2().
        config_dir = str(config_path.parent.resolve())
        config_name = config_path.name

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
            sam_model = build_sam2(
                config_file=config_name,
                ckpt_path=str(checkpoint_path),
                device=device,
            )

        ShirtSegmenter._predictor = SAM2ImagePredictor(sam_model)
        self.predictor = ShirtSegmenter._predictor

        logger.info("SAM 2.1 model loaded successfully.")

    @log_execution_time
    def segment_shirt(self, image_path, box):
        """
        Generate a segmentation mask for the shirt.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        box : list
            Bounding box received from GroundingDINO.
            Format: [x1, y1, x2, y2]

        Returns
        -------
        numpy.ndarray
            Binary mask of the shirt.
        """

        # Read the input image from the given path.
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Unable to read image: {image_path}"
            )

        # Convert the image from OpenCV BGR format to RGB format
        # because SAM expects an RGB image.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            # Give the image to SAM before running the prediction.
            self.predictor.set_image(image)

            # Convert the bounding box to a NumPy array for SAM.
            input_box = np.array(box)

            # Predict the possible segmentation masks using the bounding box.
            masks, scores, logits = self.predictor.predict(
                box=input_box,
                multimask_output=True,
            )

        # Select the mask with the highest confidence score.
        best_index = np.argmax(scores)

        return masks[best_index]

    def save_mask(self, mask, output_path):
        """
        Save the binary segmentation mask as an image.
        """

        import cv2
        import numpy as np

        mask_image = (mask * 255).astype(np.uint8)

        cv2.imwrite(output_path, mask_image)