"""
Minimum Error Boundary Cut
"""


class BoundaryCut:

    def __init__(self):
        pass

    ####################################################################
    # COMPUTE VERTICAL BOUNDARY
    ####################################################################

    def compute_vertical_boundary(
        self,
        error_surface
    ):
        """
        Compute vertical minimum-error seam.
        """
        raise NotImplementedError

    ####################################################################
    # COMPUTE HORIZONTAL BOUNDARY
    ####################################################################

    def compute_horizontal_boundary(
        self,
        error_surface
    ):
        """
        Compute horizontal minimum-error seam.
        """
        raise NotImplementedError

    ####################################################################
    # COMPUTE COMBINED MASK
    ####################################################################

    def compute_combined_mask(
        self,
        vertical_mask,
        horizontal_mask
    ):
        """
        Combine vertical and horizontal seams.
        """
        raise NotImplementedError