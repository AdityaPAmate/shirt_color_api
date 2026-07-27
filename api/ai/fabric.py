"""
File:
    api/ai/fabric.py

Purpose:
    This module prepares a fabric image for shirt replacement.

Current Milestone:
    ----------------
    ✔ Validate all inputs
    ✔ Resize or tile the uploaded fabric
    ✔ Return a prepared fabric image

Next Milestone:
    ----------------
    Apply the shirt mask and preserve lighting.
"""

import cv2
import numpy as np
from api.ai.virtual_fabric import VirtualFabric

class FabricRenderer:
    """
    FabricRenderer is responsible for preparing the uploaded fabric
    before it is applied to the detected shirt.

    This class DOES NOT:
        - detect the shirt
        - generate the SAM mask
        - save images

    It only prepares the fabric image.

    Future milestones will extend this class to perform
    realistic shirt rendering.
    """

    def __init__(self):
        """
        Initialize helper classes.

        Currently we only initialize the
        Virtual Fabric generator.

        In future more helper classes
        will be initialized here.
        """

        self.virtual_fabric = VirtualFabric()

    ####################################################################
    # INPUT VALIDATION
    ####################################################################

    def validate_inputs(self, person_image, shirt_mask, fabric_image):
        """
        Validate every input before processing.

        Parameters
        ----------
        person_image : numpy.ndarray
            Original image uploaded by the user.

        shirt_mask : numpy.ndarray
            Binary mask returned by SAM.

        fabric_image : numpy.ndarray
            Uploaded fabric texture.

        Raises
        ------
        ValueError
            If any input is invalid.
        """

        if person_image is None:
            raise ValueError("Person image is None.")

        if shirt_mask is None:
            raise ValueError("Shirt mask is None.")

        if fabric_image is None:
            raise ValueError("Fabric image is None.")

        if person_image.size == 0:
            raise ValueError("Person image is empty.")

        if shirt_mask.size == 0:
            raise ValueError("Shirt mask is empty.")

        if fabric_image.size == 0:
            raise ValueError("Fabric image is empty.")

        # The SAM mask should have the same height and width
        # as the original image.
        if person_image.shape[:2] != shirt_mask.shape[:2]:
            raise ValueError(
                "Mask size does not match person image size."
            )

    ####################################################################
    # RESIZE FABRIC
    ####################################################################

    def resize_fabric(self, fabric_image, target_height, target_width):
        """
        Resize the uploaded fabric to match the person's image size.

        Parameters
        ----------
        fabric_image : numpy.ndarray

        target_height : int

        target_width : int

        Returns
        -------
        numpy.ndarray
            Resized fabric image.
        """

        resized = cv2.resize(
            fabric_image,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )

        return resized

    ####################################################################
    # COVER FABRIC
    ####################################################################

    def cover_fabric(
            self,
            fabric_image,
            target_height,
            target_width
    ):
        """
        Resize the fabric while keeping its original aspect ratio.

        If the resized fabric becomes larger than the required size,
        crop the center portion.

        This works well for:

        - Printed fabrics
        - Large check designs
        - Floral patterns
        - Fabric photos

        Unlike tiling, this method does not repeat the pattern.
        """

        fabric_height, fabric_width = fabric_image.shape[:2]

        # ----------------------------------------------------------
        # Calculate scaling factor.
        #
        # We choose the larger scale so the entire shirt area is
        # covered.
        # ----------------------------------------------------------

        scale = max(
            target_width / fabric_width,
            target_height / fabric_height
        )

        new_width = int(fabric_width * scale)
        new_height = int(fabric_height * scale)

        resized = cv2.resize(
            fabric_image,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------------
        # Crop from the center.
        # ----------------------------------------------------------

        start_x = (new_width - target_width) // 2
        start_y = (new_height - target_height) // 2

        cropped = resized[
                  start_y:start_y + target_height,
                  start_x:start_x + target_width
                  ]

        return cropped

    ####################################################################
    # TILE FABRIC
    ####################################################################

    def tile_fabric(self, fabric_image, target_height, target_width):
        """
        Repeat (tile) the uploaded fabric until it covers
        the required output size.

        This is useful when the uploaded fabric image is
        a small texture sample.

        Example

            200x200 texture

                ↓

            Repeat horizontally and vertically

                ↓

            Large fabric sheet

                ↓

            Crop to exact output size
        """

        fabric_height, fabric_width = fabric_image.shape[:2]

        repeat_x = int(np.ceil(target_width / fabric_width))
        repeat_y = int(np.ceil(target_height / fabric_height))

        tiled = np.tile(
            fabric_image,
            (repeat_y, repeat_x, 1)
        )

        tiled = tiled[:target_height, :target_width]

        return tiled

    ####################################################################
    # PREPARE FABRIC
    ####################################################################

    def prepare_fabric(
            self,
            fabric_image,
            target_width,
            target_height,
            repeat_size=None
    ):
        """
        Prepare the fabric before rendering.

        Current Architecture
        --------------------

        This function no longer performs resizing
        or tiling itself.

        It delegates the work to the
        VirtualFabric class.

        This keeps the renderer focused only on
        rendering while VirtualFabric becomes
        responsible for cloth generation.
        """

        return self.virtual_fabric.generate(
            fabric_image=fabric_image,
            target_width=target_width,
            target_height=target_height,
            repeat_size=repeat_size
        )

    ####################################################################
    # PRESERVE ORIGINAL SHIRT LIGHTING
    ####################################################################

    ####################################################################
    # PRESERVE LIGHTING USING A NORMALIZED LIGHTING MAP
    ####################################################################

    def preserve_lighting(
            self,
            person_image,
            prepared_fabric
    ):
        """
        Preserve folds, wrinkles and shadows without copying the
        original shirt color.

        Instead of replacing the fabric brightness, we calculate
        a lighting map from the original image and multiply the
        fabric by that lighting map.

        This keeps the fabric colours much closer to the uploaded
        fabric while still showing the original lighting.
        """

        # ----------------------------------------------------------
        # Convert both images to float.
        #
        # Float images avoid rounding errors during multiplication.
        # ----------------------------------------------------------

        person = person_image.astype(np.float32)
        fabric = prepared_fabric.astype(np.float32)

        # ----------------------------------------------------------
        # Convert the original image to grayscale.
        #
        # We only need brightness information.
        # ----------------------------------------------------------

        gray = cv2.cvtColor(
            person_image,
            cv2.COLOR_BGR2GRAY
        ).astype(np.float32)

        # ----------------------------------------------------------
        # Blur the grayscale image.
        #
        # A Gaussian blur removes tiny texture details and keeps
        # only the large lighting variations such as folds and
        # shadows.
        # ----------------------------------------------------------

        illumination = cv2.GaussianBlur(
            gray,
            (31, 31),
            0
        )

        # ----------------------------------------------------------
        # Normalize the lighting map.
        #
        # Around 1.0 means no change.
        #
        # Dark folds become values below 1.
        # Bright highlights become values above 1.
        # ----------------------------------------------------------

        illumination = illumination / (illumination.mean() + 1e-6)

        # ----------------------------------------------------------
        # Limit the lighting values.
        #
        # This prevents extremely dark or bright pixels from
        # destroying the fabric colours.
        # ----------------------------------------------------------

        illumination = np.clip(
            illumination,
            0.75,
            1.25
        )

        # ----------------------------------------------------------
        # Expand the lighting map from
        #
        # (H,W)
        #
        # to
        #
        # (H,W,3)
        #
        # so it can be multiplied with a colour image.
        # ----------------------------------------------------------

        illumination = illumination[:, :, np.newaxis]

        # ----------------------------------------------------------
        # Apply lighting to the fabric.
        # ----------------------------------------------------------

        result = fabric * illumination

        # ----------------------------------------------------------
        # Prevent pixel values from going outside
        # the valid image range.
        # ----------------------------------------------------------

        result = np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

        return result

    ####################################################################
    # EXTRACT STRUCTURE MAP
    ####################################################################

    def extract_structure_map(
            self,
            person_image,
            shirt_mask
    ):
        """
        Extract only the large structural information from the shirt.

        This function intentionally ignores most of the fabric texture.

        Goal
        ----
        Keep

        - folds
        - sleeve edges
        - collar folds
        - button shadows
        - shirt placket

        Ignore

        - checks
        - stripes
        - original fabric print
        """

        # ----------------------------------------------------------
        # Convert image to grayscale.
        # ----------------------------------------------------------

        gray = cv2.cvtColor(
            person_image,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------------
        # Remove everything outside the shirt.
        # ----------------------------------------------------------

        gray = cv2.bitwise_and(
            gray,
            gray,
            mask=shirt_mask.astype(np.uint8)
        )

        # ----------------------------------------------------------
        # Large blur.
        #
        # This keeps only large lighting information.
        # ----------------------------------------------------------

        blur = cv2.GaussianBlur(
            gray,
            (31, 31),
            0
        )

        # ----------------------------------------------------------
        # Local contrast.
        #
        # Wrinkles and seams become visible.
        # Repeated fabric patterns become much weaker.
        # ----------------------------------------------------------

        structure = cv2.subtract(
            gray,
            blur
        )

        # ----------------------------------------------------------
        # Normalize result.
        # ----------------------------------------------------------

        structure = cv2.normalize(
            structure,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        return structure

    ####################################################################
    # APPLY STRUCTURE MAP
    ####################################################################

    def apply_structure_map(
            self,
            fabric_image,
            structure_map,
            strength=0.20
    ):
        """
        Add only a small amount of structural information.

        The structure map is blended with low opacity so the
        original shirt pattern is not recreated.
        """

        structure = cv2.cvtColor(
            structure_map,
            cv2.COLOR_GRAY2BGR
        ).astype(np.float32)

        fabric = fabric_image.astype(np.float32)

        structure = structure / 255.0

        result = fabric * (
                1.0 + strength * structure
        )

        result = np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

        return result

    ####################################################################
    # EXTRACT HIGH FREQUENCY DETAILS
    ####################################################################

    def extract_detail_layer(
            self,
            person_image
    ):
        """
        Extract fine details from the original shirt.

        Examples of details:

        - Buttons
        - Collar edges
        - Sleeve fold edges
        - Shirt stitching
        - Wrinkles
        - Pocket edges

        This function removes the smooth lighting component and
        keeps only the fine structures.
        """

        # ----------------------------------------------------------
        # Convert image to float.
        # Float calculations avoid clipping during subtraction.
        # ----------------------------------------------------------

        image = person_image.astype(np.float32)

        # ----------------------------------------------------------
        # Create a blurred version of the image.
        #
        # The blur represents the low-frequency information
        # (lighting and smooth colour transitions).
        # ----------------------------------------------------------

        blurred = cv2.GaussianBlur(
            image,
            (21, 21),
            0
        )

        # ----------------------------------------------------------
        # High-frequency details are obtained by subtracting the
        # blurred image from the original image.
        # ----------------------------------------------------------

        detail = image - blurred

        return detail

    ####################################################################
    # APPLY DETAIL LAYER
    ####################################################################

    def apply_detail_layer(
            self,
            fabric_image,
            detail_layer,
            strength=1.0
    ):
        """
        Overlay the extracted detail layer onto the rendered fabric.

        Parameters
        ----------
        fabric_image : numpy.ndarray
            Fabric after lighting preservation.

        detail_layer : numpy.ndarray
            Fine details extracted from the original shirt.

        strength : float
            Controls how much detail is added back.
            1.0 = original strength
            0.5 = softer
            1.5 = stronger
        """

        fabric = fabric_image.astype(np.float32)

        result = fabric + detail_layer * strength

        result = np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

        return result

    ####################################################################
    # APPLY SHIRT MASK
    ####################################################################

    def apply_shirt_mask(
            self,
            person_image,
            shirt_mask,
            prepared_fabric
    ):
        """
        Apply the shirt mask to the prepared fabric.

        Purpose
        -------
        This function copies the fabric only inside the detected
        shirt region.

        Everything outside the shirt remains exactly the same.

        NOTE
        ----
        This milestone does NOT preserve lighting or folds.
        That will be implemented in the next milestone.
        """

        # --------------------------------------------------------------
        # Convert the SAM mask into a binary mask.
        #
        # Background = 0
        # Shirt = 255
        # --------------------------------------------------------------
        binary_mask = (shirt_mask > 0).astype("uint8") * 255

        # --------------------------------------------------------------
        # Convert the single-channel mask into a 3-channel mask.
        #
        # Person Image : (H, W, 3)
        # Fabric Image : (H, W, 3)
        # Mask         : (H, W)
        #
        # We need:
        # Mask         : (H, W, 3)
        # --------------------------------------------------------------
        binary_mask = cv2.merge([
            binary_mask,
            binary_mask,
            binary_mask
        ])

        # --------------------------------------------------------------
        # Copy the original image.
        #
        # We never modify the original image directly.
        # --------------------------------------------------------------
        output = person_image.copy()

        # --------------------------------------------------------------
        # Wherever the shirt mask is white,
        # replace those pixels with the prepared fabric.
        # --------------------------------------------------------------
        output[binary_mask == 255] = prepared_fabric[binary_mask == 255]

        return output

    ####################################################################
    # COMPLETE FABRIC RENDER
    ####################################################################

    def render(
            self,
            person_image,
            shirt_mask,
            fabric_image,
            fabric_mode="tile"
    ):
        """
        Complete fabric rendering pipeline.

        Steps
        -----
        1. Validate inputs.
        2. Prepare fabric.
        3. Preserve original lighting.
        4. Apply shirt mask.
        """

        # Step 1
        prepared_fabric = self.prepare_fabric(
            fabric_image=fabric_image,
            target_width=person_image.shape[1],
            target_height=person_image.shape[0],
            repeat_size=None
        )

        # Step 2
        # ----------------------------------------------------------
        # Preserve overall lighting.
        # ----------------------------------------------------------

        realistic_fabric = self.preserve_lighting(
            person_image,
            prepared_fabric
        )

        # ----------------------------------------------------------
        # Extract fine details from the original shirt.
        # ----------------------------------------------------------

        # ----------------------------------------------------------
        # Extract shirt structure.
        # ----------------------------------------------------------

        structure_map = self.extract_structure_map(
            person_image,
            shirt_mask
        )

        # ----------------------------------------------------------
        # Blend the structure into the new fabric.
        # ----------------------------------------------------------

        realistic_fabric = self.apply_structure_map(
            realistic_fabric,
            structure_map,
            strength=0.20
        )

        # ----------------------------------------------------------
        # Replace only the shirt region.
        # ----------------------------------------------------------

        output = self.apply_shirt_mask(
            person_image,
            shirt_mask,
            realistic_fabric
        )

        return output