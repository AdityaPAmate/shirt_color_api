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
import math

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

            "pattern_repeat": self.detect_pattern_repeat(fabric_image),

            "pattern_direction":  self.detect_pattern_direction(fabric_image),

            "is_seamless": None

        }

    ####################################################################
    # DETECT PATTERN DIRECTION
    ####################################################################

    def detect_pattern_direction(
            self,
            fabric_image
    ):
        """
        Detect the dominant direction of the fabric pattern.

        Current Version
        ----------------
        We use image gradients to estimate whether the
        fabric mainly contains

        - vertical lines
        - horizontal lines
        - diagonal lines

        Future versions may use frequency-domain
        analysis (FFT) or deep learning.

        Parameters
        ----------
        fabric_image : numpy.ndarray

        Returns
        -------
        str

            horizontal
            vertical
            diagonal
            unknown
        """

        # ----------------------------------------------------------
        # Convert to grayscale.
        #
        # Gradient detection works on a single channel.
        # ----------------------------------------------------------

        gray = cv2.cvtColor(
            fabric_image,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------------
        # Calculate horizontal and vertical gradients.
        #
        # Sobel measures intensity change.
        # ----------------------------------------------------------

        grad_x = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )

        grad_y = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )

        # ----------------------------------------------------------
        # Compute average gradient strength.
        # ----------------------------------------------------------

        mean_x = np.mean(np.abs(grad_x))
        mean_y = np.mean(np.abs(grad_y))

        # ----------------------------------------------------------
        # Compare gradient strengths.
        # ----------------------------------------------------------

        difference = abs(mean_x - mean_y)

        # If both are very similar,
        # assume diagonal or mixed pattern.

        if difference < 5:
            return "diagonal"

        if mean_x > mean_y:
            return "vertical"

        return "horizontal"

    ####################################################################
    # DETECT PATTERN REPEAT
    ####################################################################

    def detect_pattern_repeat(
            self,
            fabric_image
    ):
        """
        Estimate the repeating distance of the fabric pattern.

        Current Version
        ----------------
        We estimate the repeat using autocorrelation of the
        grayscale image.

        Returns
        -------
        int

            Estimated repeat distance in pixels.

        Returns None if no reliable repeat is detected.
        """

        # ----------------------------------------------------------
        # Convert image to grayscale.
        # ----------------------------------------------------------

        gray = cv2.cvtColor(
            fabric_image,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------------
        # Take the middle row.
        #
        # This keeps the first implementation simple.
        # ----------------------------------------------------------

        row = gray[
            gray.shape[0] // 2
            ].astype(np.float32)

        # ----------------------------------------------------------
        # Remove brightness offset.
        #
        # We only want repeating structure.
        # ----------------------------------------------------------

        row -= np.mean(row)

        # ----------------------------------------------------------
        # Calculate autocorrelation.
        # ----------------------------------------------------------

        correlation = np.correlate(
            row,
            row,
            mode="full"
        )

        correlation = correlation[
                      correlation.size // 2:
                      ]

        # ----------------------------------------------------------
        # Ignore the first peak.
        #
        # Lag = 0 is always the maximum.
        # ----------------------------------------------------------

        correlation[0] = 0

        # ----------------------------------------------------------
        # Find strongest remaining peak.
        # ----------------------------------------------------------

        repeat = np.argmax(correlation)

        # ----------------------------------------------------------
        # Ignore unrealistic values.
        # ----------------------------------------------------------

        if repeat < 8:
            return None

        return int(repeat)