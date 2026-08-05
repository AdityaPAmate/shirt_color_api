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
generate a larger piece of cloth by tiling it.

CHANGELOG
---------
NEW: generate() completely rewritten.

     Old version #1 (active) only center-cropped the fabric and
     never actually tiled it -> caused stretching/incorrect sizing
     whenever fabric_info["original_fabric"] size did not already
     match the person image.

     Old version #2 (commented out) called self.image_quilter,
     which was never created in __init__ -> would have crashed
     if it were ever un-commented.

     New version performs PERIOD-ALIGNED SEAMLESS TILING:
       1. Scale the fabric so the pattern size looks right on
          the shirt (reuses existing compute_scale_factor /
          resize_for_tiling logic).
       2. Crop the fabric down to an exact whole-number multiple
          of the detected pattern repeat (both X and Y), so the
          tile boundary always falls in the "empty" background
          area of the pattern instead of cutting a flower/check
          in half. This is what removes the visible seam lines.
       3. np.tile() the cropped block until it covers the full
          target canvas, then crop to exact size.

REMOVED: make_seamless() (np.roll offset trick). It only *moved*
         the seam to the center of the image, it never removed it.
         Period-aligned cropping (step 2 above) makes it unnecessary.
"""

import cv2
import numpy as np
from pathlib import Path


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
    # CREATE VIRTUAL FABRIC (Period-aligned seamless tiling)
    ####################################################################

    def generate(
            self,
            fabric_image,
            target_width,
            target_height,
            repeat_size=None,
            repeat_size_y=None
    ):
        """
        Generate a large virtual fabric using period-aligned seamless
        tiling.

        Parameters
        ----------
        fabric_image : ndarray
            The uploaded fabric photo (BGR, uint8).

        target_width : int
            Width of the person image / shirt canvas.

        target_height : int
            Height of the person image / shirt canvas.

        repeat_size : int or None
            Detected horizontal (X-axis) pattern repeat, in pixels,
            coming from FabricAnalyzer.detect_pattern_repeat().
            None -> fabric is plain / repeat not detected, no
            period-alignment is applied on X.

        repeat_size_y : int or None
            Detected vertical (Y-axis) pattern repeat, in pixels,
            coming from FabricAnalyzer.detect_pattern_repeat_y().
            None -> no period-alignment is applied on Y.

        Returns
        -------
        ndarray
            (target_height, target_width, 3) tiled fabric image.
        """

        BASE_DIR = Path(__file__).resolve().parents[2]
        DEBUG_FOLDER = BASE_DIR / "test_images" / "debug"
        DEBUG_FOLDER.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug0_before_scale.png"),
            fabric_image
        )

        # ------------------------------------------------------------
        # Step A: Scale the pattern to a sensible size relative to
        # the shirt (existing, already-tested logic).
        # ------------------------------------------------------------

        scale_factor = self.compute_scale_factor(
            fabric_image,
            repeat_size
        )

        fabric_image = self.resize_for_tiling(
            fabric_image,
            scale_factor
        )

        # The detected repeat distances must be scaled by the same
        # factor, otherwise period-alignment in Step B will use a
        # stale (pre-resize) period value.
        if repeat_size:
            repeat_size = max(1, int(repeat_size * scale_factor))

        if repeat_size_y:
            repeat_size_y = max(1, int(repeat_size_y * scale_factor))

        print(f"Scale factor applied : {scale_factor}")
        print(f"Repeat size (x) after scaling : {repeat_size}")
        print(f"Repeat size (y) after scaling : {repeat_size_y}")

        # ------------------------------------------------------------
        # Step B: Crop the fabric down to an exact whole-number
        # multiple of the pattern repeat, on both axes.
        #
        # This guarantees the tile boundary lands in the background
        # gap between motifs instead of cutting a motif in half,
        # which is what causes a visible seam line.
        #
        # If no repeat was detected (plain fabric, or detection
        # failed), the fabric is used as-is on that axis - a plain
        # fabric has no motif to misalign, so this is safe.
        # ------------------------------------------------------------

        h, w = fabric_image.shape[:2]

        if repeat_size and repeat_size > 0 and repeat_size <= w:
            crop_w = (w // repeat_size) * repeat_size
        else:
            crop_w = w

        if repeat_size_y and repeat_size_y > 0 and repeat_size_y <= h:
            crop_h = (h // repeat_size_y) * repeat_size_y
        else:
            crop_h = h

        crop_w = max(crop_w, 1)
        crop_h = max(crop_h, 1)

        fabric_image = fabric_image[0:crop_h, 0:crop_w]

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug1_period_aligned_tile_unit.png"),
            fabric_image
        )

        print(f"Period-aligned tile unit size (w x h) : {crop_w} x {crop_h}")

        # ------------------------------------------------------------
        # Step C: Tile the period-aligned unit across the full
        # target canvas, then crop to the exact requested size.
        # ------------------------------------------------------------

        tiles_y = int(np.ceil(target_height / fabric_image.shape[0])) + 1
        tiles_x = int(np.ceil(target_width / fabric_image.shape[1])) + 1

        tiled = np.tile(fabric_image, (tiles_y, tiles_x, 1))

        tiled = tiled[0:target_height, 0:target_width]

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug2_final_tiled_fabric.png"),
            tiled
        )

        return tiled