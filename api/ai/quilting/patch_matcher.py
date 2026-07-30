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
"""
Patch Matching
"""

import numpy as np


class PatchMatcher:

    def __init__(self):
        pass

    ####################################################################
    # CALCULATE OVERLAP ERROR
    ####################################################################

    def calculate_overlap_error(
        self,
        candidate_patch,
        existing_region,
        overlap
    ):
        """
        Calculate SSD error inside the overlap region.

        Smaller error means the candidate patch matches
        the already generated fabric better.
        """

        if existing_region is None:
            return 0.0

        # Height overlap
        if existing_region.shape[0] == overlap:

            diff = (
                candidate_patch[:overlap].astype(np.float32)
                - existing_region.astype(np.float32)
            )

        # Width overlap
        elif existing_region.shape[1] == overlap:

            diff = (
                candidate_patch[:, :overlap].astype(np.float32)
                - existing_region.astype(np.float32)
            )

        else:
            raise ValueError(
                "Invalid overlap region shape."
            )

        return float(np.sum(diff * diff))

    ####################################################################
    # CALCULATE FULL OVERLAP ERROR
    ####################################################################

    def calculate_full_overlap_error(
            self,
            candidate_patch,
            left_overlap=None,
            top_overlap=None,
            overlap=16
    ):
        """
        Calculate the total SSD error from both
        the left and top overlap regions.
        """

        total_error = 0.0

        # Left overlap
        if left_overlap is not None:
            diff = (
                    candidate_patch[:, :overlap].astype(np.float32)
                    - left_overlap.astype(np.float32)
            )

            total_error += np.sum(diff * diff)

        # Top overlap
        if top_overlap is not None:
            diff = (
                    candidate_patch[:overlap].astype(np.float32)
                    - top_overlap.astype(np.float32)
            )

            total_error += np.sum(diff * diff)

        return float(total_error)

    ####################################################################
    # FIND BEST PATCH
    ####################################################################

    def find_best_patch(
            self,
            candidates,
            left_overlap=None,
            top_overlap=None,
            overlap=16
    ):
        """
        Select the patch with the minimum SSD overlap error.

        Parameters
        ----------
        candidates : list[np.ndarray]

        existing_region : np.ndarray

        overlap : int

        Returns
        -------
        np.ndarray
        """

        if len(candidates) == 0:
            raise ValueError("No candidate patches found.")

        best_patch = candidates[0]
        best_error = float("inf")

        for patch in candidates:

            error = self.calculate_full_overlap_error(
                candidate_patch=patch,
                left_overlap=left_overlap,
                top_overlap=top_overlap,
                overlap=overlap
            )

            if error < best_error:
                best_error = error
                best_patch = patch

        return best_patch