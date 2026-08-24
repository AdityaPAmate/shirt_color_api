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
NEW: person_image is now read EARLIER (right after Step 1), because
     Step 2.6 (bbox from garment_mask) needs person_image.shape to
     clamp the padded bbox -- it used to be read later in Step 3,
     which caused an "unresolved reference" (used before assignment)
     error. Step 3 no longer re-reads it.
NEW: Step 4 now passes box=fabric_fit_box (the garment_mask-based
     box) instead of box=detection["box"] -- passing the old shirt
     box here was defeating the whole point of Step 2.6.
"""

import os
import cv2
import numpy as np
import psutil
import threading
import time
import logging

from api.ai.detector import ShirtDetector
from api.ai.segmenter import ShirtSegmenter
from api.ai.fabric import FabricRenderer
from api.ai.fabric_analyzer import FabricAnalyzer
from api.ai.pattern_scale_estimator import PatternScaleEstimator
from api.ai.garment_template import GarmentTemplate
from pathlib import Path
from api.ai.utils import log_execution_time

logger = logging.getLogger(__name__)


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
    def _get_process_memory_mb(self):
        """
        Return total RAM currently used by this Python process.
        """
        process = psutil.Process(os.getpid())

        memory_bytes = process.memory_info().rss

        return memory_bytes / (1024 * 1024)


    def _start_memory_monitor(self):
        """
        Continuously monitor this Python process and store
        the highest RAM usage during pipeline execution.
        """

        self._memory_monitor_running = True

        self._peak_memory_mb = self._get_process_memory_mb()

        def monitor():
            while self._memory_monitor_running:

                current_memory_mb = self._get_process_memory_mb()

                if current_memory_mb > self._peak_memory_mb:
                    self._peak_memory_mb = current_memory_mb

                time.sleep(0.1)

        self._memory_monitor_thread = threading.Thread(
            target=monitor,
            daemon=True
        )

        self._memory_monitor_thread.start()


    def _stop_memory_monitor(self):
        """
        Stop monitoring and return peak RAM usage.
        """

        self._memory_monitor_running = False

        self._memory_monitor_thread.join()

        return self._peak_memory_mb
    # ============================================================
    # INITIALIZE AI MODELS
    # ============================================================
    def __init__(self):
        """
        Load every model only once.

        GroundingDINO and SAM are expensive models.

        Therefore we initialize them only once.
        """

        logger.info("Loading AI models...")

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

        logger.info("All AI models loaded successfully.")

        # ========================================================
        # MEMORY AFTER ALL MODELS ARE LOADED
        # ========================================================

        models_memory_mb = self._get_process_memory_mb()

        logger.info(
            "RAM used after loading AI models: %.2f MB",
            models_memory_mb
        )

    ####################################################################
    # FABRIC REPLACEMENT PIPELINE
    ####################################################################
    @log_execution_time
    def replace_fabric(
        self,
        person_image_path,
        fabric_image_path,
        output_path,
        garment_type="shirt",
        detection_target="shirt"
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

        detection_target : str
            GroundingDINO la denyacha text prompt -- "shirt", "pant",
            "jacket", "kurta" etc. Konta garment DETECT karायcha te
            control karto (garment_type mask-TEMPLATE control karto --
            donhi vegळे, pan sadhya sathi sarkhich value denyat yeईl).

        Returns
        -------
        str
            Output image path.
        """

        BASE_DIR = Path(__file__).resolve().parents[2]

        DEBUG_FOLDER = BASE_DIR / "test_images" / "debug"

        DEBUG_FOLDER.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # START WHOLE PIPELINE MEMORY MONITORING
        # ============================================================

        pipeline_start_memory_mb = self._get_process_memory_mb()

        self._start_memory_monitor()

        logger.info(
            "Starting fabric replacement pipeline | "
            "Garment type: %s | Detection target: %s | RAM before pipeline: %.2f MB",
            garment_type,
            detection_target,
            pipeline_start_memory_mb
        )

        ############################################################
        # STEP 1
        ############################################################

        logger.info(
            "Step 1: Starting garment detection | Target: %s",
            detection_target
        )

        detection = self.detector.detect_shirt(
            person_image_path,
            detection_target=detection_target
        )

        logger.info(
            "Step 1 completed | Box: %s | Confidence: %s",
            detection["box"],
            detection["confidence"]
        )

        # ----------------------------------------------------------
        # NEW (moved up): read person_image here, right after
        # detection, instead of later in Step 3. Step 2.6 (below)
        # needs person_image.shape to clamp the padded bbox, so it
        # must exist before Step 2.6 runs.
        # ----------------------------------------------------------
        person_image = cv2.imread(person_image_path)

        if person_image is None:
            raise ValueError(
                f"Unable to read person image : {person_image_path}"
            )

        # Save detected shirt bounding box for debugging
        debug_detection = person_image.copy()

        x1, y1, x2, y2 = map(int, detection["box"])

        cv2.rectangle(
            debug_detection,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug_0.1_detected_shirt.png"),
            debug_detection
        )

        ############################################################
        # STEP 2
        ############################################################

        logger.info("Step 2: Starting shirt mask generation")

        shirt_mask = self.segmenter.segment_shirt(
            person_image_path,
            detection["box"]
        )

        logger.info("Step 2 completed: Mask Generateration successfully")

        debug_mask = (shirt_mask > 0).astype("uint8") * 255

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug_0.2_shirt_mask.png"),
            debug_mask
        )

        ############################################################
        # STEP 2.5 (NEW) : Shirt Mask -> Requested Garment Mask
        ############################################################

        logger.info(
            "Step 2.5: Preparing garment mask | Garment type: %s",
            garment_type
        )

        garment_mask = self.garment_template.get_garment_mask(
            shirt_mask,
            garment_type=garment_type
        )

        logger.info(
            "Step 2.5 completed: Garment mask is ready"
        )

        cv2.imwrite(
            str(DEBUG_FOLDER/"debug_garment_mask.png" ),
            (garment_mask.astype("uint8") * 255) if garment_mask.dtype == bool else garment_mask
        )

        # ============================================================
        # STEP 2.6: Get the correct bounding box from the garment mask.
        #
        # detection["box"] is the original GroundingDINO shirt box.
        # For a kurta, the garment mask may extend below or outside
        # the original shirt box.
        #
        # Get a new bounding box from the non-zero pixels of the
        # garment mask. Add an 8-pixel margin on all sides and keep
        # the box inside the person image boundaries.
        #
        # person_image was already loaded after Step 1, so its shape
        # can be used here.
        # ============================================================

        mask_ys, mask_xs = np.where(garment_mask > 0)

        # Tight bounding box with no extra margin.
        tight_x1 = int(mask_xs.min())
        tight_y1 = int(mask_ys.min())
        tight_x2 = int(mask_xs.max())
        tight_y2 = int(mask_ys.max())

        # Extra margin in pixels to add on all sides.
        BBOX_MARGIN = 8

        # Add the margin and keep the box inside the person image boundaries.
        person_h, person_w = person_image.shape[:2]

        padded_x1 = max(0, tight_x1 - BBOX_MARGIN)
        padded_y1 = max(0, tight_y1 - BBOX_MARGIN)
        padded_x2 = min(person_w - 1, tight_x2 + BBOX_MARGIN)
        padded_y2 = min(person_h - 1, tight_y2 + BBOX_MARGIN)

        fabric_fit_box = [
            float(padded_x1),
            float(padded_y1),
            float(padded_x2),
            float(padded_y2)
        ]

        logger.info(
            "Garment mask box ready | Tight box: %s | Fabric box with %d px margin: %s",
            [tight_x1, tight_y1, tight_x2, tight_y2],
            BBOX_MARGIN,
            fabric_fit_box
        )

        ############################################################
        # STEP 3
        ############################################################

        logger.info("Step 3: Reading images and analyzing fabric")

        # person_image was already loaded after Step 1.
        # Do not read it again here.

        fabric_image = cv2.imread(fabric_image_path)

        # ----------------------------------------------------------
        # Analyze the uploaded fabric.
        #
        # The returned information can be used for:
        #
        # - Virtual Fabric
        # - Panel Cutting
        # - Panel Warping
        # ----------------------------------------------------------

        fabric_info = self.fabric_analyzer.analyze(
            fabric_image
        )

        # ----------------------------------------------------------
        # Estimate the pattern scale.
        # ----------------------------------------------------------

        logger.info(
            "Fabric pattern repeat: %s",
            fabric_info["pattern_repeat"]
        )

        scale_info = self.pattern_scale_estimator.estimate(
            fabric_info
        )

        logger.info(
            "Pattern scale information: %s",
            scale_info
        )

        logger.info(
            "Fabric information: %s",
            fabric_info
        )

        if fabric_image is None:
            raise ValueError(
                f"Unable to read fabric image : {fabric_image_path}"
            )

        logger.info(
            "Step 3 completed | Person image shape: %s | Fabric image shape: %s",
            person_image.shape,
            fabric_image.shape
        )

        ############################################################
        # STEP 4
        ############################################################

        logger.info(
            "Step 4: Starting fabric rendering | Garment type: %s",
            garment_type
        )

        result = self.fabric_renderer.render(
            person_image=person_image,
            shirt_mask=garment_mask,
            fabric_info=fabric_info,
            garment_type=garment_type,
            box=fabric_fit_box
        )

        logger.info("Step 4 completed: Fabric rendering finished")

        ############################################################
        # STEP 5
        ############################################################

        logger.info(
            "Step 5: Saving output | Output path: %s",
            output_path
        )

        output_folder = os.path.dirname(output_path)

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

        cv2.imwrite(
            output_path,
            result
        )

        logger.info(
            "Step 5 completed: Output saved successfully | Output path: %s",
            output_path
        )

        # ============================================================
        # STOP PIPELINE MEMORY MONITORING
        # ============================================================

        peak_memory_mb = self._stop_memory_monitor()

        pipeline_end_memory_mb = self._get_process_memory_mb()

        extra_memory_mb = (
                peak_memory_mb - pipeline_start_memory_mb
        )

        # ============================================================
        # FINAL MEMORY REPORT
        # ============================================================

        logger.info(
            "Memory report | RAM before: %.2f MB | Peak RAM: %.2f MB | "
            "RAM after: %.2f MB | Extra RAM used: %.2f MB",
            pipeline_start_memory_mb,
            peak_memory_mb,
            pipeline_end_memory_mb,
            extra_memory_mb
        )

        logger.info(
            "Pipeline completed successfully | Output path: %s",
            output_path
        )

        return output_path