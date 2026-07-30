"""
==========================================================
Virtual Fabric
==========================================================

Purpose
-------
This module creates a large virtual cloth from the uploaded
fabric image.

Important
---------
We DO NOT stretch the uploaded fabric.

Instead, we preserve the original pattern scale and
generate a larger piece of cloth.

Future versions will also preserve

- repeat alignment
- fabric orientation
- garment panels
"""

import cv2
import numpy as np


class VirtualFabric:

    def __init__(self):
        pass

    ####################################################################
    # RESIZE FABRIC BEFORE TILING
    ####################################################################

    def resize_for_tiling(
            self,
            fabric_image,
            scale_factor=1.0
    ):
        """
        Resize the uploaded fabric before creating the virtual cloth.

        scale_factor > 1.0
            Pattern becomes larger.

        scale_factor < 1.0
            Pattern becomes smaller.

        scale_factor = 1.0
            Keep original size.
        """

        if scale_factor <= 0:
            raise ValueError("scale_factor must be greater than zero.")

        if abs(scale_factor - 1.0) < 0.001:
            return fabric_image

        h, w = fabric_image.shape[:2]

        new_w = max(1, int(w * scale_factor))
        new_h = max(1, int(h * scale_factor))

        return cv2.resize(
            fabric_image,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )

    ####################################################################
    # COMPUTE SCALE FACTOR
    ####################################################################

    def compute_scale_factor(
            self,
            fabric_image,
            repeat_size
    ):
        """
        Returns the resize factor that should be applied before tiling.
        """

        if repeat_size is None:
            return 1.0

        image_width = fabric_image.shape[1]

        # Repeat occupies a very large portion of the image.
        if repeat_size > image_width * 0.40:
            return 0.50

        # Large repeat
        if repeat_size > image_width * 0.25:
            return 0.70

        # Medium repeat
        if repeat_size > image_width * 0.15:
            return 0.85

        # Small repeat
        return 1.0

    ####################################################################
    # CREATE SEAMLESS FABRIC
    ####################################################################

    def make_seamless(
            self,
            fabric_image
    ):
        """
        Reduce visible tile boundaries before creating
        the virtual cloth.
        """

        h, w = fabric_image.shape[:2]

        offset_x = w // 2
        offset_y = h // 2

        shifted = np.roll(
            fabric_image,
            shift=(-offset_y, -offset_x),
            axis=(0, 1)
        )

        return shifted
    ####################################################################
    # CREATE VIRTUAL FABRIC
    ####################################################################

    def generate(
        self,
        fabric_image,
        target_width,
        target_height,
        repeat_size=None
    ):
        """
        Generate a large virtual fabric.

        Parameters
        ----------
        fabric_image : ndarray

        target_width : int

        target_height : int

        repeat_size : int

            Reserved for future improvements.

        Returns
        -------
        ndarray
        """

        # ----------------------------------------------------------
        # Temporary V1.
        # Later this value will come from PatternScaleEstimator.
        # ----------------------------------------------------------

        scale_factor = self.compute_scale_factor(
            fabric_image,
            repeat_size
        )

        fabric_image = self.resize_for_tiling(
            fabric_image,
            scale_factor
        )

        fabric_image = self.make_seamless(
            fabric_image
        )

        fabric_h, fabric_w = fabric_image.shape[:2]

        # ----------------------------------------------------------
        # Future: Pattern-aware scaling.
        #
        # Currently we only receive the detected repeat size.
        #
        # In future milestones this value will be converted into
        # a scale factor before generating the virtual fabric.
        #
        # For now we simply validate it so the complete pipeline
        # becomes pattern-aware.
        # ----------------------------------------------------------

        if repeat_size is not None:

            if repeat_size <= 0:
                raise ValueError(
                    "Pattern repeat must be greater than zero."
                )

            print(f"Detected Pattern Repeat : {repeat_size} pixels")

            

        # ----------------------------------------------------------
        # Calculate how many repetitions are required.
        #
        # We intentionally repeat the ORIGINAL fabric.
        #
        # We DO NOT resize it.
        # ----------------------------------------------------------

        repeat_x = int(np.ceil(target_width / fabric_w)) + 2
        repeat_y = int(np.ceil(target_height / fabric_h)) + 2

        # ----------------------------------------------------------
        # Build a large virtual cloth.
        # ----------------------------------------------------------

        virtual = np.tile(
            fabric_image,
            (repeat_y, repeat_x, 1)
        )

        # ----------------------------------------------------------
        # Crop only the required size.
        #
        # Notice:
        # We crop.
        #
        # We do NOT resize.
        # ----------------------------------------------------------

        virtual = virtual[
            0:target_height,
            0:target_width
        ]

        return virtual