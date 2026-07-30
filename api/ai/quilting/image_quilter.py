"""
Image Quinter

Coordinates the complete Image Quilting pipeline.

This class does not implement the algorithm itself.

It only coordinates:

- Patch Extraction
- Patch Matching
- Boundary Cut
- Patch Composition
"""

from .patch_extractor import PatchExtractor
from .patch_matcher import PatchMatcher
from .boundary_cut import BoundaryCut
from .patch_compositor import PatchCompositor


class ImageQuilter:

    def __init__(self):

        self.patch_extractor = PatchExtractor()

        self.patch_matcher = PatchMatcher()

        self.boundary_cut = BoundaryCut()

        self.patch_compositor = PatchCompositor()

    ####################################################################
    # GENERATE
    ####################################################################

    def generate(
        self,
        fabric_image,
        target_width,
        target_height,
        patch_size,
        overlap
    ):
        """
        Generate a quilted virtual fabric.

        Implementation will be added in IQ-008.
        """
        raise NotImplementedError