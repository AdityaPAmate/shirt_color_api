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
from pathlib import Path

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
        Preserve the original shirt lighting while keeping the uploaded
        fabric colours almost unchanged.

        This version extracts only the illumination component and
        applies it to the uploaded fabric.
        """

        # ---------------------------------------------
        # Convert images to float
        # ---------------------------------------------
        fabric = prepared_fabric.astype(np.float32)

        gray = cv2.cvtColor(
            person_image,
            cv2.COLOR_BGR2GRAY
        ).astype(np.float32)

        # ---------------------------------------------
        # Estimate illumination
        # ---------------------------------------------
        illumination = cv2.GaussianBlur(
            gray,
            (81, 81),
            0
        )

        # ---------------------------------------------
        # Normalize illumination
        # ---------------------------------------------
        illumination = illumination / (illumination.mean() + 1e-6)

        # ---------------------------------------------
        # Keep stronger lighting variation
        # ---------------------------------------------
        illumination = np.clip(
            illumination,
            0.60,
            1.40
        )

        illumination = illumination[:, :, np.newaxis]

        # ---------------------------------------------
        # Apply lighting
        # ---------------------------------------------
        result = fabric * illumination

        result = np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

        return result

    def extract_fold_map(
            self,
            person_image,
            shirt_mask
    ):
        """
        Extract only folds and wrinkles.

        This removes most of the original shirt texture.
        """

        gray = cv2.cvtColor(
            person_image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.bitwise_and(
            gray,
            gray,
            mask=shirt_mask.astype(np.uint8)
        )

        # Large illumination
        low_frequency = cv2.GaussianBlur(
            gray,
            (41, 41),
            0
        )

        # High-pass image
        high_frequency = cv2.subtract(
            gray,
            low_frequency
        )

        # Normalize
        fold_map = cv2.normalize(
            high_frequency,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        return fold_map

    def clean_fold_map(
            self,
            fold_map
    ):
        """
        Remove tiny shirt texture while keeping folds.
        """

        kernel = np.ones((5, 5), np.uint8)

        cleaned = cv2.morphologyEx(
            fold_map,
            cv2.MORPH_OPEN,
            kernel
        )

        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel
        )

        return cleaned

    def apply_fold_map(
            self,
            fabric_image,
            fold_map,
            strength=0.18
    ):
        """
        Apply only fold information.

        Original shirt colour and pattern are not copied.
        """

        fold = fold_map.astype(np.float32)

        fold = (fold - 128.0) / 255.0

        fold = fold[:, :, np.newaxis]

        fabric = fabric_image.astype(np.float32)

        result = fabric + (fold * 255 * strength)

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
            fabric_info,
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
        # ----------------------------------------------------------
        # Extract analyzed fabric information.
        # ----------------------------------------------------------

        # ----------------------------------------------------------
        # Debug Output Folder
        # ----------------------------------------------------------

        BASE_DIR = Path(__file__).resolve().parents[2]

        DEBUG_FOLDER = BASE_DIR / "test_images" / "debug"

        DEBUG_FOLDER.mkdir(parents=True, exist_ok=True)



        fabric_image = fabric_info["original_fabric"]
        cv2.imwrite(
            str(DEBUG_FOLDER / "debug_0_prepared_fabric2.png"),
            fabric_image
        )


        pattern_repeat = fabric_info["pattern_repeat"]

        print("Before prepare:", fabric_image.shape)
        print("Fabric dtype:", fabric_image.dtype)

        # Step 1
        # prepared_fabric = self.prepare_fabric(
        #     fabric_image=fabric_image,
        #     target_width=person_image.shape[1],
        #     target_height=person_image.shape[0],
        #     repeat_size=pattern_repeat
        # )

        prepared_fabric=fabric_image

        print("After prepare:", prepared_fabric.shape)
        print("prepared dtype:", prepared_fabric.dtype)

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug_1_prepared_fabric2.png"),
            prepared_fabric
        )
        # Step 2
        # ----------------------------------------------------------
        # Preserve overall lighting.
        # ----------------------------------------------------------

        realistic_fabric = self.preserve_lighting(
            person_image,
            prepared_fabric
        )

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug_2_after_lighting2.png"),
            realistic_fabric
        )

        # ----------------------------------------------------------
        # Extract fine details from the original shirt.
        # ----------------------------------------------------------

        # ----------------------------------------------------------
        # Extract shirt structure.
        # ----------------------------------------------------------

        fold_map = self.extract_fold_map(
            person_image,
            shirt_mask
        )

        fold_map = self.clean_fold_map(
            fold_map
        )

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug_3_after_fold_map2.png"),
            fold_map
        )

        # ----------------------------------------------------------
        # Blend the structure into the new fabric.
        # ----------------------------------------------------------

        # realistic_fabric = self.apply_structure_map(
        #     realistic_fabric,
        #     structure_map,
        #     strength=0.35
        # )

        realistic_fabric = self.apply_fold_map(
            realistic_fabric,
            fold_map,
            strength=0.18
        )

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug_4_after_realastic_fabric2.png"),
            realistic_fabric
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