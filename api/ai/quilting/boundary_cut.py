"""
Minimum Error Boundary Cut
"""

import numpy as np


class BoundaryCut:

    def __init__(self):
        pass

    ####################################################################
    # COMPUTE VERTICAL SEAM
    ####################################################################

    def compute_vertical_seam(
        self,
        error_map
    ):
        """
        Find the minimum-cost vertical seam using Dynamic Programming.

        Parameters
        ----------
        error_map : ndarray (H, W)

        Returns
        -------
        list[int]

            Column index for each row.
        """

        height, width = error_map.shape

        cost = error_map.astype(np.float32).copy()
        parent = np.zeros((height, width), dtype=np.int32)

        # Forward DP
        for y in range(1, height):

            for x in range(width):

                left = max(0, x - 1)
                right = min(width - 1, x + 1)

                previous = cost[y - 1, left:right + 1]

                offset = np.argmin(previous)

                parent[y, x] = left + offset

                cost[y, x] += previous[offset]

        # Backtrack
        seam = [0] * height

        seam[-1] = int(np.argmin(cost[-1]))

        for y in range(height - 2, -1, -1):
            seam[y] = parent[y + 1, seam[y + 1]]

        return seam

    ####################################################################
    # CREATE SEAM MASK
    ####################################################################

    def create_mask(
        self,
        seam,
        width
    ):
        """
        Convert seam into a binary mask.
        """

        height = len(seam)

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        for y, x in enumerate(seam):
            mask[y, :x] = 1

        return mask