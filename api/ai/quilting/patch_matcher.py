"""
Patch Matching
"""


class PatchMatcher:

    def __init__(self):
        pass

    ####################################################################
    # CALCULATE OVERLAP ERROR
    ####################################################################

    def calculate_overlap_error(
        self,
        candidate_patch,
        existing_region
    ):
        """
        Calculate SSD overlap error.
        """
        raise NotImplementedError

    ####################################################################
    # FIND BEST PATCH
    ####################################################################

    def find_best_patch(
        self,
        candidates,
        existing_region
    ):
        """
        Return the best matching patch.
        """
        raise NotImplementedError