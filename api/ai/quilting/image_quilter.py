"""
Image Quilter

Main controller for Image Quilting.
"""

import numpy as np

from .patch_extractor import PatchExtractor
from .patch_matcher import PatchMatcher
from .patch_compositor import PatchCompositor
import numpy as np
import random


class ImageQuilter:

    def __init__(self):

        self.extractor = PatchExtractor()
        self.matcher = PatchMatcher()
        self.compositor = PatchCompositor()

    ####################################################################
    # SELECT CANDIDATE PATCH
    ####################################################################

    def select_candidate(
            self,
            candidates,
            existing_region,
            overlap,
            tolerance=1.10
    ):
        """
        Instead of always selecting the single best patch,
        randomly choose one from the good candidates.

        This produces more natural-looking textures.
        """

        scored = []

        for patch in candidates:
            error = self.matcher.calculate_overlap_error(
                patch,
                existing_region,
                overlap
            )

            scored.append((error, patch))

        scored.sort(key=lambda x: x[0])

        best_error = scored[0][0]

        threshold = best_error * tolerance

        good_candidates = [
            patch
            for error, patch in scored
            if error <= threshold
        ]

        return random.choice(good_candidates)

    ####################################################################
    # GENERATE TEXTURE
    ####################################################################

    def generate(
            self,
            fabric_image,
            output_width,
            output_height,
            patch_size=64,
            overlap=16
    ):
        """
        Generate a larger fabric texture using
        Image Quilting.
        """

        canvas = np.zeros(
            (
                output_height,
                output_width,
                3
            ),
            dtype=np.uint8
        )

        locations = self.extractor.get_candidate_locations(
            fabric_image,
            patch_size
        )

        candidates = []

        for px, py in locations:
            patch = self.extractor.extract_patch(
                fabric_image,
                px,
                py,
                patch_size
            )

            candidates.append(patch)

        step = patch_size - overlap

        for y in range(0, output_height - patch_size + 1, step):

            for x in range(0, output_width - patch_size + 1, step):

                ####################################################
                # First Patch
                ####################################################

                if x == 0 and y == 0:
                    seed_patch = random.choice(candidates)

                    canvas[
                    0:patch_size,
                    0:patch_size
                    ] = seed_patch

                    continue

                ####################################################
                # Collect overlap regions
                ####################################################

                left_overlap = None
                top_overlap = None

                if x > 0:
                    left_overlap = canvas[
                                   y:y + patch_size,
                                   x:x + overlap
                                   ]

                if y > 0:
                    top_overlap = canvas[
                                  y:y + overlap,
                                  x:x + patch_size
                                  ]

                ####################################################
                # Find Best Patch
                ####################################################

                patch = self.matcher.find_best_patch(
                    candidates=candidates,
                    left_overlap=left_overlap,
                    top_overlap=top_overlap,
                    overlap=overlap
                )

                ####################################################
                # Blend Patch
                ####################################################

                self.compositor.blend_left_overlap(
                    canvas=canvas,
                    patch=patch,
                    x=x,
                    y=y,
                    overlap=overlap
                )

        return canvas
