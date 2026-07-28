import cv2
import numpy as np


class FabricScaler:
    """
    Estimates an appropriate scaling factor for the uploaded fabric.

    Goal:
    -----
    Prevent large fabric photos from appearing stretched
    when applied to a shirt.
    """

    def estimate_scale(self, fabric_image):
        """
        Returns a scaling factor.

        Returns
        -------
        float
            Example:
            0.35
            0.45
            0.60
            1.00
        """

        gray = cv2.cvtColor(fabric_image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 80, 180)

        edge_density = np.count_nonzero(edges) / edges.size

        print("\nFabric Edge Density :", edge_density)

        # Large patterns -> shrink more
        if edge_density < 0.03:
            return 0.35

        elif edge_density < 0.06:
            return 0.50

        elif edge_density < 0.10:
            return 0.70

        # Small patterns -> keep almost original
        return 1.0