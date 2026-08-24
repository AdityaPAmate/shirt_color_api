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

CHANGELOG
---------
NEW: Added detect_pattern_repeat_y() -> vertical (Y-axis) repeat
     period detection, needed for period-aligned seamless tiling
     in VirtualFabric.generate().
NEW: analyze() now also returns "pattern_repeat_y".
"""

import logging

import cv2
import numpy as np
import math
from api.ai.utils import log_execution_time

logger = logging.getLogger(__name__)


@log_execution_time
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

        # lab = cv2.cvtColor(
        #     fabric_image,
        #     cv2.COLOR_BGR2LAB
        # )
        #
        # l, a, b = cv2.split(lab)
        #
        # # ------------------------------------------
        # # Improve brightness only.
        # # ------------------------------------------
        #
        # clahe = cv2.createCLAHE(
        #     clipLimit=2.5,
        #     tileGridSize=(8, 8)
        # )
        #
        # l = clahe.apply(l)
        #
        # lab = cv2.merge((l, a, b))
        #
        # normalized = cv2.cvtColor(
        #     lab,
        #     cv2.COLOR_LAB2BGR
        # )
        #
        # # ------------------------------------------
        # # Small denoising.
        # # ------------------------------------------
        #
        # normalized = cv2.fastNlMeansDenoisingColored(
        #     normalized,
        #     None,
        #     3,
        #     3,
        #     7,
        #     21
        # )
        #
        # return normalized

        return fabric_image

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

        normalized_fabric = self.normalize_fabric(
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

        has_pattern = self.has_repeating_pattern(fabric_image)

        pattern_repeat = None
        pattern_repeat_y = None  # NEW

        if has_pattern:
            pattern_repeat = self.detect_pattern_repeat(fabric_image)
            pattern_repeat_y = self.detect_pattern_repeat_y(fabric_image)  # NEW

        return {

            "original_fabric": fabric_image,
            "normalized_fabric": normalized_fabric,

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

            "has_pattern": has_pattern,

            "pattern_repeat": pattern_repeat,

            "pattern_repeat_y": pattern_repeat_y,  # NEW

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
    # DETECT PATTERN REPEAT (HORIZONTAL / X-AXIS)
    ####################################################################

    def detect_pattern_repeat(
            self,
            fabric_image
    ):
        """
        Estimate the dominant repeating distance of the fabric pattern
        along the X-axis (horizontal).

        Current Version
        ----------------
        Uses averaged autocorrelation across multiple rows instead
        of only the middle row.
        """

        gray = cv2.cvtColor(
            fabric_image,
            cv2.COLOR_BGR2GRAY
        )

        gray = gray.astype(np.float32)

        height, width = gray.shape

        # ----------------------------------------------------------
        # Sample multiple rows.
        # ----------------------------------------------------------

        sample_rows = np.linspace(
            int(height * 0.20),
            int(height * 0.80),
            9,
            dtype=int
        )

        accumulated = np.zeros(width, dtype=np.float32)

        valid_rows = 0

        for row_index in sample_rows:
            row = gray[row_index].copy()

            row -= np.mean(row)

            correlation = np.correlate(
                row,
                row,
                mode="full"
            )

            correlation = correlation[
                          correlation.size // 2:
                          ]

            correlation[0] = 0

            accumulated += correlation

            valid_rows += 1

        if valid_rows == 0:
            return None

        correlation = accumulated / valid_rows

        # ----------------------------------------------------------
        # Reject fabrics with very weak repeating signals.
        # Plain fabrics should not report a pattern repeat.
        # ----------------------------------------------------------

        energy = np.std(gray)

        if energy < 12:
            return None

        # ----------------------------------------------------------
        # Ignore very small repeats.
        # ----------------------------------------------------------

        MIN_REPEAT = 20

        correlation[:MIN_REPEAT] = 0

        # ----------------------------------------------------------
        # Find local peaks.
        # ----------------------------------------------------------

        peaks = []

        for i in range(
                MIN_REPEAT,
                len(correlation) - 1
        ):

            if (
                    correlation[i] > correlation[i - 1]
                    and
                    correlation[i] > correlation[i + 1]
                    and
                    correlation[i] > (correlation.max() * 0.75)
            ):
                peaks.append(i)

        if not peaks:
            return None

        # ----------------------------------------------------------
        # Choose the largest significant repeat.
        # ----------------------------------------------------------

        threshold = correlation.max() * 0.75

        significant = sorted(peaks)

        if not significant:
            return None

        logger.debug("Detected peaks: %s", significant)
        logger.debug("Selected repeat: %s", max(significant))

        # This function used to require at least 2 peaks before
        # accepting a repeat value -- so a strong, confident single
        # peak was wrongly rejected as None (e.g. on a small crop,
        # or when a second peak fell just below the 0.75 threshold).
        # The energy check above (std < 12 -> reject) and the
        # peak-height check (correlation.max() * 0.75) already filter
        # out weak/spurious peaks, so a single confident peak is now
        # accepted as trustworthy on its own.
        return int(significant[-1])

    ####################################################################
    # DETECT PATTERN REPEAT (VERTICAL / Y-AXIS)
    ####################################################################
    # NEW METHOD
    ####################################################################

    def detect_pattern_repeat_y(
            self,
            fabric_image
    ):
        """
        Same idea as detect_pattern_repeat(), but runs along columns
        instead of rows -> detects the VERTICAL repeat period.

        This is required for period-aligned seamless tiling
        (VirtualFabric.generate()), because floral / check fabrics
        repeat in BOTH the X and Y directions.
        """

        gray = cv2.cvtColor(
            fabric_image,
            cv2.COLOR_BGR2GRAY
        ).astype(np.float32)

        height, width = gray.shape

        # ----------------------------------------------------------
        # Sample multiple columns.
        # ----------------------------------------------------------

        sample_cols = np.linspace(
            int(width * 0.20),
            int(width * 0.80),
            9,
            dtype=int
        )

        accumulated = np.zeros(height, dtype=np.float32)

        valid_cols = 0

        for col_index in sample_cols:
            col = gray[:, col_index].copy()

            col -= np.mean(col)

            correlation = np.correlate(
                col,
                col,
                mode="full"
            )

            correlation = correlation[
                          correlation.size // 2:
                          ]

            correlation[0] = 0

            accumulated += correlation

            valid_cols += 1

        if valid_cols == 0:
            return None

        correlation = accumulated / valid_cols

        # ----------------------------------------------------------
        # Reject fabrics with very weak repeating signals.
        # ----------------------------------------------------------

        energy = np.std(gray)

        if energy < 12:
            return None

        # ----------------------------------------------------------
        # Ignore very small repeats.
        # ----------------------------------------------------------

        MIN_REPEAT = 20

        correlation[:MIN_REPEAT] = 0

        # ----------------------------------------------------------
        # Find local peaks.
        # ----------------------------------------------------------

        peaks = []

        for i in range(
                MIN_REPEAT,
                len(correlation) - 1
        ):

            if (
                    correlation[i] > correlation[i - 1]
                    and
                    correlation[i] > correlation[i + 1]
                    and
                    correlation[i] > (correlation.max() * 0.75)
            ):
                peaks.append(i)

        if not peaks:
            return None

        significant = sorted(peaks)

        if not significant:
            return None

        logger.debug("Detected Y-axis peaks: %s", significant)
        logger.debug("Selected Y-axis repeat: %s", max(significant))

        # Same reasoning as detect_pattern_repeat(): the "at least 2
        # peaks required" rule has been removed. A single confident
        # peak (one that already passed the energy and peak-height
        # thresholds above) is now accepted.
        return int(significant[-1])

    ####################################################################
    # HAS REPEATING PATTERN
    ####################################################################

    def has_repeating_pattern(
            self,
            fabric_image
    ):
        """
        Returns True if the fabric contains a visible repeating pattern.
        Returns False for plain or nearly plain fabrics.
        """

        gray = cv2.cvtColor(
            fabric_image,
            cv2.COLOR_BGR2GRAY
        )

        edges = cv2.Canny(
            gray,
            80,
            160
        )

        edge_ratio = np.count_nonzero(edges) / edges.size

        logger.debug("Edge ratio: %.4f", edge_ratio)

        return edge_ratio > 0.03

    def detect_pattern_repeat_fft_fallback(self, gray_crop, min_repeat=8, max_repeat=80):
        """
        Fallback used when the autocorrelation (2-peak) method fails.
        Works even on small crops, because it only looks for the
        single strongest (dominant) frequency instead of requiring
        two clear peaks.
        """
        f = np.fft.fft2(gray_crop.astype(np.float32) - gray_crop.mean())
        fshift = np.abs(np.fft.fftshift(f))

        h, w = gray_crop.shape
        cy, cx = h // 2, w // 2

        # Zero out the region near DC (0 frequency)
        fshift[cy - 2:cy + 3, cx - 2:cx + 3] = 0

        # Find the strongest remaining frequency
        max_idx = np.unravel_index(np.argmax(fshift), fshift.shape)
        fy, fx = max_idx
        dist = np.hypot(fy - cy, fx - cx)

        if dist < 1:
            return None

        period = min(h, w) / dist

        if min_repeat <= period <= max_repeat:
            return int(round(period))

        return None