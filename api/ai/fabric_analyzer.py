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
    # NORMALIZE FABRIC
    ####################################################################

    def normalize_fabric(
            self,
            fabric_image
    ):
        """
        Normalize the uploaded fabric.

        Goal
        ----
        Different users capture fabric from different
        distances and lighting conditions.

        This function prepares the fabric before it is
        converted into a virtual cloth.

        Current Version

        ✔ Remove lighting variation

        ✔ Improve contrast

        ✔ Keep original colours

        ✔ Reduce camera noise

        Future

        • Perspective correction

        • Pattern alignment

        • Cloth rectification
        """

        # ------------------------------------------
        # Convert to LAB colour space.
        #
        # LAB separates brightness from colours.
        # ------------------------------------------

        lab = cv2.cvtColor(
            fabric_image,
            cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)

        # ------------------------------------------
        # Improve brightness only.
        # ------------------------------------------

        clahe = cv2.createCLAHE(
            clipLimit=2.5,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))

        normalized = cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2BGR
        )

        # ------------------------------------------
        # Small denoising.
        # ------------------------------------------

        normalized = cv2.fastNlMeansDenoisingColored(
            normalized,
            None,
            3,
            3,
            7,
            21
        )

        return normalized
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
        # Normalize the uploaded fabric before extracting
        # any information from it.
        # ----------------------------------------------------------

        fabric_image = self.normalize_fabric(
            fabric_image
        )

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

            "normalized_fabric": fabric_image,

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

    ####################################################################
    # DETECT PATTERN REPEAT
    ####################################################################

    def detect_pattern_repeat(
            self,
            fabric_image
    ):
        """
        Estimate the dominant repeating distance of the fabric pattern.

        Current Strategy
        ----------------
        1. Convert the image to grayscale.
        2. Analyse multiple horizontal rows instead of only one row.
        3. Compute autocorrelation for every selected row.
        4. Average all autocorrelation curves.
        5. Ignore very small lag values.
        6. Detect local peaks.
        7. Return the largest visually meaningful repeat.

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

        height, width = gray.shape

        # ----------------------------------------------------------
        # Select multiple rows across the fabric.
        #
        # Using multiple rows is much more reliable than analysing
        # only the centre row.
        # ----------------------------------------------------------

        number_of_rows = min(9, height)

        row_indices = np.linspace(
            int(height * 0.10),
            int(height * 0.90),
            number_of_rows,
            dtype=int
        )

        autocorrelations = []

        # ----------------------------------------------------------
        # Compute autocorrelation for every selected row.
        # ----------------------------------------------------------

        for row_index in row_indices:

            row = gray[row_index].astype(np.float32)

            # Remove brightness offset.
            row -= np.mean(row)

            # Skip rows with almost no texture.
            if np.std(row) < 1:
                continue

            correlation = np.correlate(
                row,
                row,
                mode="full"
            )

            correlation = correlation[
                          correlation.size // 2:
                          ]

            # Normalise so every row contributes equally.
            if correlation[0] != 0:
                correlation = correlation / correlation[0]

            autocorrelations.append(correlation)

        # ----------------------------------------------------------
        # No usable rows.
        # ----------------------------------------------------------

        if len(autocorrelations) == 0:
            return None

        # ----------------------------------------------------------
        # Average autocorrelation.
        # ----------------------------------------------------------

        average_correlation = np.mean(
            autocorrelations,
            axis=0
        )

        # ----------------------------------------------------------
        # Ignore lag zero.
        # ----------------------------------------------------------

        average_correlation[0] = 0

        # ----------------------------------------------------------
        # Ignore very small lag values.
        #
        # Small lags usually correspond to fine texture rather than
        # the main fabric pattern.
        # ----------------------------------------------------------

        minimum_lag = max(8, width // 40)

        average_correlation[:minimum_lag] = 0

        # ----------------------------------------------------------
        # Detect local peaks.
        # ----------------------------------------------------------

        peaks = []

        for i in range(
                minimum_lag,
                len(average_correlation) - 1
        ):

            if (
                    average_correlation[i] >
                    average_correlation[i - 1]
                    and
                    average_correlation[i] >
                    average_correlation[i + 1]
            ):
                peaks.append(i)

        # ----------------------------------------------------------
        # No peaks found.
        # ----------------------------------------------------------

        if len(peaks) == 0:
            return None

        # ----------------------------------------------------------
        # Keep only significant peaks.
        #
        # A peak must be at least 30% of the strongest peak.
        # ----------------------------------------------------------

        peak_values = [
            average_correlation[p]
            for p in peaks
        ]

        strongest_peak = max(peak_values)

        significant_peaks = [

            p

            for p in peaks

            if average_correlation[p] >= strongest_peak * 0.30

        ]

        if len(significant_peaks) == 0:
            return None

        # ----------------------------------------------------------
        # Return the largest visually meaningful repeat.
        # ----------------------------------------------------------

        return int(max(significant_peaks))