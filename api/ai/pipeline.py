"""
File:
    api/ai/pipeline.py

Purpose
-------
This file acts as the main AI pipeline.

Instead of calling:
    - GroundingDINO
    - SAM
    - Fabric Renderer

from different places every time,
we call only ONE function from this file.

Advantages
----------
1. Easy testing
2. Easy Django API integration
3. Future frontend integration
4. Reusable architecture
5. Single responsibility

CHANGELOG
---------
NEW: Added garment_type support ("shirt" / "kurta").
     SAM still detects/segments the SHIRT only (detection logic
     unchanged). After segmentation, GarmentTemplate converts the
     shirt mask into the requested garment's mask (Kurta = shirt
     mask extended downward). Fabric rendering then runs on
     whichever mask corresponds to garment_type.
"""

import os
import cv2

from api.ai.detector import ShirtDetector
from api.ai.segmenter import ShirtSegmenter
from api.ai.fabric import FabricRenderer
from api.ai.fabric_analyzer import FabricAnalyzer
from api.ai.pattern_scale_estimator import PatternScaleEstimator
from api.ai.garment_template import GarmentTemplate
from pathlib import Path


class ShirtPipeline:
    """
    Main AI Pipeline.

    This class connects all AI modules together.

    Current Pipeline

        Person Image
              │
              ▼
        GroundingDINO
              │
              ▼
          SAM 2.1  (Shirt Mask)
              │
              ▼
      Garment Template  (Shirt Mask -> Shirt/Kurta Mask)
              │
              ▼
      Fabric Rendering
              │
              ▼
         Final Output

    Future Features

    - Shirt Recolor
    - Logo Placement
    - Print Placement
    - Pattern Alignment
    """

    def __init__(self):
        """
        Load every model only once.

        GroundingDINO and SAM are expensive models.

        Therefore we initialize them only once.
        """

        print("\nLoading AI Models...")

        self.detector = ShirtDetector()

        self.segmenter = ShirtSegmenter()

        # ----------------------------------------------------------
        # Analyze the uploaded fabric.
        # This class only studies the fabric.
        # It does not modify any image.
        # ----------------------------------------------------------
        self.fabric_analyzer = FabricAnalyzer()

        self.pattern_scale_estimator = PatternScaleEstimator()

        # ----------------------------------------------------------
        # Responsible for rendering the fabric onto the shirt.
        # ----------------------------------------------------------
        self.fabric_renderer = FabricRenderer()

        # ----------------------------------------------------------
        # NEW: Converts the SAM shirt mask into the requested
        # garment's mask (currently: shirt as-is, or kurta).
        # ----------------------------------------------------------
        self.garment_template = GarmentTemplate()

        print("All AI Models Loaded Successfully.")

    ####################################################################
    # FABRIC REPLACEMENT PIPELINE
    ####################################################################

    def replace_fabric(
        self,
        person_image_path,
        fabric_image_path,
        output_path,
        fabric_mode="tile",
        garment_type="shirt"
    ):
        """
        Complete fabric replacement pipeline.

        Parameters
        ----------
        person_image_path : str
            Path of person image.

        fabric_image_path : str
            Path of uploaded fabric.

        output_path : str
            Output image path.

        fabric_mode : str
            "tile" -> Repeat fabric pattern.
            "fit"  -> Stretch fabric.

        garment_type : str
            "shirt" -> use SAM shirt mask as-is.
            "kurta" -> extend SAM shirt mask into a basic Kurta mask.

        Returns
        -------
        str
            Output image path.
        """

        BASE_DIR = Path(__file__).resolve().parents[2]

        DEBUG_FOLDER = BASE_DIR / "test_images" / "debug"

        DEBUG_FOLDER.mkdir(parents=True, exist_ok=True)

        print("\n====================================")
        print("Starting Fabric Replacement Pipeline")
        print("====================================")
        print("Garment Type :", garment_type)

        ############################################################
        # STEP 1
        ############################################################

        print("\nStep 1 : Detect Shirt")

        detection = self.detector.detect_shirt(
            person_image_path
        )

        print("Detection Completed")
        print("Bounding Box :", detection["box"])
        print("Confidence   :", detection["confidence"])

        ############################################################
        # STEP 2
        ############################################################

        print("\nStep 2 : Generate Shirt Mask")

        shirt_mask = self.segmenter.segment_shirt(
            person_image_path,
            detection["box"]
        )

        print("Mask Generated Successfully")

        ############################################################
        # STEP 2.5 (NEW) : Shirt Mask -> Requested Garment Mask
        ############################################################

        print("\nStep 2.5 : Prepare Garment Mask (", garment_type, ")")

        garment_mask = self.garment_template.get_garment_mask(
            shirt_mask,
            garment_type=garment_type
        )

        print("Garment Mask Ready")

        cv2.imwrite(
            str(DEBUG_FOLDER/"debug_garment_mask.png" ),
            (garment_mask.astype("uint8") * 255) if garment_mask.dtype == bool else garment_mask
        )

        ############################################################
        # STEP 3
        ############################################################

        print("\nStep 3 : Read Images")

        person_image = cv2.imread(person_image_path)

        fabric_image = cv2.imread(fabric_image_path)

        # ----------------------------------------------------------
        # Analyse the uploaded fabric.
        #
        # The returned information will be used in future milestones
        # such as:
        #
        # - Virtual Fabric
        # - Panel Cutting
        # - Panel Warping
        # ----------------------------------------------------------

        fabric_info = self.fabric_analyzer.analyze(
            fabric_image
        )

        # ----------------------------------------------------------
        # Estimate pattern scale.
        #
        # Current version only prepares the architecture.
        # ----------------------------------------------------------
        print(fabric_info["pattern_repeat"])

        scale_info = self.pattern_scale_estimator.estimate(
            fabric_info
        )

        print("\n========== Pattern Scale ==========")

        for key, value in scale_info.items():
            print(f"{key} : {value}")

        print("===================================\n")

        # Use the normalized fabric returned by FabricAnalyzer
        fabric_image = cv2.imread(fabric_image_path)

        print("\n========== Fabric Information ==========")

        for key, value in fabric_info.items():
            print(f"{key} : {value}")

        print("========================================\n")

        if person_image is None:
            raise ValueError(
                f"Unable to read person image : {person_image_path}"
            )

        if fabric_image is None:
            raise ValueError(
                f"Unable to read fabric image : {fabric_image_path}"
            )

        print("Person Image Shape :", person_image.shape)
        print("Fabric Image Shape :", fabric_image.shape)

        ############################################################
        # STEP 4
        ############################################################

        print("\nStep 4 : Render Fabric")

        result = self.fabric_renderer.render(
            person_image=person_image,
            shirt_mask=garment_mask,
            fabric_info=fabric_info,
            garment_type= garment_type
        )

        print("Fabric Rendering Completed")

        ############################################################
        # STEP 5
        ############################################################

        print("\nStep 5 : Save Output")

        output_folder = os.path.dirname(output_path)

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

        cv2.imwrite(
            output_path,
            result
        )

        print("Output Saved Successfully")

        print("\nOutput Path")
        print(output_path)

        print("\nPipeline Completed Successfully.")

        return output_path