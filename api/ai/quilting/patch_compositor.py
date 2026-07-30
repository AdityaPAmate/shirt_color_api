"""
Patch Compositor

Responsible for blending patches together.
"""
import cv2
import numpy as np

from .boundary_cut import BoundaryCut


class PatchCompositor:

    def __init__(self):

        self.boundary_cut = BoundaryCut()

    ####################################################################
    # CREATE ERROR MAP
    ####################################################################

    def create_error_map(
        self,
        existing_overlap,
        candidate_overlap
    ):
        """
        Compute pixel-wise squared error.
        """

        diff = (
            existing_overlap.astype(np.float32)
            - candidate_overlap.astype(np.float32)
        )

        return np.sum(diff * diff, axis=2)

    ####################################################################
    # BLEND LEFT OVERLAP
    ####################################################################

    def blend_left_overlap(
        self,
        canvas,
        patch,
        x,
        y,
        overlap
    ):
        """
        Blend a patch with the existing canvas using
        minimum error boundary cut.
        """

        if x == 0:
            canvas[
                y:y + patch.shape[0],
                x:x + patch.shape[1]
            ] = patch

            return

        existing_overlap = canvas[
            y:y + patch.shape[0],
            x:x + overlap
        ]

        candidate_overlap = patch[:, :overlap]

        error_map = self.create_error_map(
            existing_overlap,
            candidate_overlap
        )

        seam = self.boundary_cut.compute_vertical_seam(
            error_map
        )

        mask = self.boundary_cut.create_mask(
            seam,
            overlap
        )

        patch = patch.copy()

        patch[:, :overlap][mask == 1] = existing_overlap[
            mask == 1
        ]

        canvas[
            y:y + patch.shape[0],
            x:x + patch.shape[1]
        ] = patch