"""
==========================================================
Fabric Analyzer
==========================================================

Purpose
-------
This module is responsible for analysing the uploaded
fabric image.

It DOES NOT perform fabric rendering.

It DOES NOT modify the shirt.

It only studies the uploaded fabric and returns useful
information that will be used by future modules.

Future modules

Fabric Analyzer
        ↓
Virtual Fabric
        ↓
Garment Template
        ↓
Panel Warper
        ↓
Garment Assembler

Keeping this logic in a separate class follows the
Single Responsibility Principle (SRP).

Only one responsibility:
Analyse fabric.
"""

import cv2
import numpy as np


class FabricAnalyzer:
    """
    Analyse the uploaded fabric image.

    Current Version
    ----------------
    This first version only collects basic information.

    Future versions will analyse

    • pattern repeat
    • pattern direction
    • pattern scale
    • seamless texture
    • dominant colours
    """

    def __init__(self):
        """
        Constructor.

        No model is loaded here.

        Future:
        If we use any AI model for analysing fabric,
        it can be loaded only once inside this class.
        """
        pass

    ####################################################################
    # ANALYSE FABRIC
    ####################################################################

    def analyze(
        self,
        fabric_image
    ):
        """
        Analyse the uploaded fabric image.

        Parameters
        ----------
        fabric_image : numpy.ndarray

            OpenCV image.

        Returns
        -------
        dict

            Dictionary containing all analysed properties.
        """

        if fabric_image is None:
            raise ValueError("Fabric image is None.")

        # ----------------------------------------------------------
        # Image Size
        # ----------------------------------------------------------

        height, width = fabric_image.shape[:2]

        # ----------------------------------------------------------
        # Aspect Ratio
        # ----------------------------------------------------------

        aspect_ratio = width / height

        # ----------------------------------------------------------
        # Mean Colour
        #
        # This is not currently used for rendering.
        #
        # It will be useful later for
        #
        # • quality checking
        # • exposure checking
        # • colour consistency
        # ----------------------------------------------------------

        mean_bgr = cv2.mean(fabric_image)[:3]

        # ----------------------------------------------------------
        # Return analysed information.
        # ----------------------------------------------------------

        return {

            "height": height,

            "width": width,

            "aspect_ratio": aspect_ratio,

            "mean_bgr": mean_bgr,

            # --------------------------------------------------
            # Future values.
            #
            # These will be calculated in later milestones.
            # --------------------------------------------------

            "pattern_scale": None,

            "pattern_repeat": None,

            "pattern_direction": None,

            "is_seamless": None

        }