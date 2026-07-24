"""
recolor.py

Purpose
-------
Recolor only the segmented shirt while preserving
its natural folds, shadows, and lighting.
"""

import cv2
import numpy as np


class ShirtRecolor:

    # Hue values in OpenCV HSV (0-179)
    COLOR_MAP = {
        "red": 0,
        "orange": 15,
        "yellow": 30,
        "green": 60,
        "cyan": 90,
        "blue": 120,
        "purple": 140,
        "pink": 165,
    }

    def recolor(self, image_path, mask, color_name):
        """
        Recolor the shirt.

        Parameters
        ----------
        image_path : str
            Path to original image.

        mask : numpy.ndarray
            SAM segmentation mask.

        color_name : str
            Target shirt color.
        """

        if color_name not in self.COLOR_MAP:
            raise ValueError(
                f"Unsupported color: {color_name}"
            )

        # Read image
        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Convert mask to boolean
        mask = mask.astype(bool)

        # Change hue only inside mask
        hsv[..., 0][mask] = self.COLOR_MAP[color_name]

        # Increase saturation
        hsv[..., 1][mask] = 200

        # IMPORTANT:
        # We DO NOT change Value (Brightness).
        # This preserves shadows and folds.

        # Convert back to BGR
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return result

    def save_image(self, image, output_path):
        """
        Save recolored image.
        """

        cv2.imwrite(output_path, image)