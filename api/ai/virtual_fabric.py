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

        fabric_h, fabric_w = fabric_image.shape[:2]

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