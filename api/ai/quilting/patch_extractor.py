"""
Patch Extraction
"""

import cv2
import numpy as np


class PatchExtractor:

    def __init__(self):
        pass

    ####################################################################
    # VALIDATE SOURCE
    ####################################################################

    def validate_source(
        self,
        fabric_image,
        patch_size
    ):
        """
        Validate that the fabric image is large enough.
        """

        if fabric_image is None:
            raise ValueError("Fabric image is None.")

        height, width = fabric_image.shape[:2]

        if height < patch_size or width < patch_size:
            raise ValueError(
                "Fabric image is smaller than patch size."
            )

    ####################################################################
    # EXTRACT PATCH
    ####################################################################

    def extract_patch(
        self,
        fabric_image,
        x,
        y,
        patch_size
    ):
        """
        Extract one square patch.
        """

        return fabric_image[
            y:y + patch_size,
            x:x + patch_size
        ].copy()

    ####################################################################
    # GET CANDIDATE LOCATIONS
    ####################################################################

    def get_candidate_locations(
        self,
        fabric_image,
        patch_size,
        stride=None
    ):
        """
        Return all valid top-left patch locations.
        """

        self.validate_source(
            fabric_image,
            patch_size
        )

        if stride is None:
            stride = max(1, patch_size // 4)

        height, width = fabric_image.shape[:2]

        locations = []

        for y in range(
            0,
            height - patch_size + 1,
            stride
        ):
            for x in range(
                0,
                width - patch_size + 1,
                stride
            ):
                locations.append((x, y))

        return locations