import logging

import cv2
import numpy as np
import os
from pathlib import Path
from api.ai.utils import log_execution_time

logger = logging.getLogger(__name__)


def xyxy_to_xywh(bbox_xyxy):
    """
    GroundingDINO's output is in (x1, y1, x2, y2) format
    (per the official Hugging Face documentation:
    top_left_x, top_left_y, bottom_right_x, bottom_right_y).

    This function converts that to (x, y, w, h), and rounds
    the float coordinates to integers for pixel indexing.
    """

    x1, y1, x2, y2 = bbox_xyxy

    x = int(round(x1))
    y = int(round(y1))
    w = int(round(x2 - x1))
    h = int(round(y2 - y1))

    return x, y, w, h


def compute_uniform_scale(fabric_w, fabric_h, bbox_w, bbox_h):
    """
    Computes the single (uniform) scale needed to fit the fabric
    completely into the bbox (no empty space left anywhere).
    Aspect ratio is preserved.
    """

    scale_x = bbox_w / fabric_w
    scale_y = bbox_h / fabric_h

    scale = max(scale_x, scale_y)

    return scale


def resize_fabric_uniform(fabric_img, scale):
    """
    Resizes the fabric by the given scale (preserving aspect ratio).
    """

    fabric_h, fabric_w = fabric_img.shape[:2]

    new_w = int(round(fabric_w * scale))
    new_h = int(round(fabric_h * scale))

    # Downsampling (scale < 1.0) uses INTER_AREA,
    # upsampling (scale >= 1.0) uses INTER_LANCZOS4
    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_LANCZOS4
    )

    resized = cv2.resize(
        fabric_img,
        (new_w, new_h),
        interpolation=interpolation
    )
    logger.info("Fabric resized, new shape: %s", resized.shape)

    return resized


def center_crop_to_bbox(resized_fabric, bbox_w, bbox_h):
    """
    Center-crops the resized fabric (which may be slightly larger
    than the bbox) down to the exact bbox size.
    """

    resized_h, resized_w = resized_fabric.shape[:2]

    start_x = (resized_w - bbox_w) // 2
    start_y = (resized_h - bbox_h) // 2

    cropped = resized_fabric[
        start_y:start_y + bbox_h,
        start_x:start_x + bbox_w
    ]

    logger.info("Fabric cropped, final shape: %s", cropped.shape)

    return cropped


def create_black_canvas(person_h, person_w, channels):
    """
    Creates a fully black canvas the same size as the person image.
    """

    canvas = np.zeros(
        (person_h, person_w, channels),
        dtype=np.uint8
    )

    return canvas


def paste_fabric_on_canvas(canvas, cropped_fabric, bbox_x, bbox_y):
    """
    Pastes the cropped fabric onto the black canvas at the bbox's
    (x, y) location.
    """
    logger.info("Placing fabric on canvas at x=%s, y=%s", bbox_x, bbox_y)
    bbox_h, bbox_w = cropped_fabric.shape[:2]

    canvas[
        bbox_y:bbox_y + bbox_h,
        bbox_x:bbox_x + bbox_w
    ] = cropped_fabric

    return canvas


def draw_bbox_for_verification(person_img_path, bbox_xywh, output_folder):
    """
    Draws the given bbox rectangle on the person image and saves it,
    so the bbox placement can be visually verified against the
    actual shirt location.

    NOTE: This function still expects a FILE PATH (string), not a
    numpy array, because it calls cv2.imread() internally. It is
    NOT called from fit_fabric_to_bbox() anymore (see below), since
    the pipeline now passes already-loaded numpy arrays, not paths.
    Keep this function for standalone/manual debugging only -- call
    it separately with an actual image path if you need the red-box
    preview image.
    """

    person_img = cv2.imread(person_img_path)

    if person_img is None:
        logger.error(
            "Could not read the person image at this path: %s",
            person_img_path
        )
        return

    bbox_x, bbox_y, bbox_w, bbox_h = bbox_xywh

    img_with_box = person_img.copy()

    cv2.rectangle(
        img_with_box,
        (bbox_x, bbox_y),
        (bbox_x + bbox_w, bbox_y + bbox_h),
        (0, 0, 255),  # red (BGR)
        thickness=3
    )

    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, "bbox_verification.png")

    cv2.imwrite(output_path, img_with_box)

    logger.info("Verification image saved here: %s", output_path)

@log_execution_time
def fit_fabric_to_bbox(
    person_image,
    fabric_image,
    groundingdino_bbox_xyxy

):
    """
    Main fabric-to-bounding-box fitting function.

    Fits the fabric into the bbox size (preserving aspect ratio),
    then pastes it onto a black canvas the same size as the person
    image, at the bbox location.

    IMPORTANT (pipeline integration change):
    person_image and fabric_image are now expected to be already
    LOADED numpy arrays (as produced by cv2.imread() earlier in
    fabric.py's render()/pipeline.py), NOT file paths. This is why
    cv2.imread() is no longer called on them inside this function --
    calling cv2.imread() on an array (instead of a string path) is
    exactly what caused the earlier crash:
        cv2.error: ... Expected 'filename' to be a str or path-like object
    """

    BASE_DIR = Path(__file__).resolve().parents[2]

    # person_image = f"{BASE_DIR}/test_images/person16.jpeg"
    # fabric_image = f"{BASE_DIR}/fabric_images/floral_fabric7.png"
    output_folder_path = f"{BASE_DIR}/test_images/bbox_fit_output"

    # Example of the actual GroundingDINO output, in (x1, y1, x2, y2) format:
    # groundingdino_bbox_xyxy = [
    #     362.3363037109375,
    #     672.457275390625,
    #     700.0819091796875,
    #     1113.4124755859375
    # ]

    bbox_xywh = xyxy_to_xywh(groundingdino_bbox_xyxy)

    # REMOVED the draw_bbox_for_verification() call that used to be here.
    # That function needs a file path (it calls cv2.imread() on it),
    # but person_image is now a numpy array -- passing the array to a
    # function expecting a path is exactly what raised the OpenCV
    # "Expected 'filename' to be a str or path-like object" error.
    # If you want the red-box debug preview, call
    # draw_bbox_for_verification() separately with an actual image path.

    # -------------------------------------------------------------------------

    # person_image and fabric_image are already numpy arrays here,
    # so we use them directly instead of calling cv2.imread() again.
    # (Previously this line was: person_img = cv2.imread(person_image),
    # which crashed because person_image was not a path.)
    person_img = person_image
    fabric_img = fabric_image

    if person_img is None:
        logger.error("Person image is None: %s", person_image)
        return

    if fabric_img is None:
        logger.error("Fabric image is None: %s", fabric_image)
        return

    person_h, person_w, channels = person_img.shape
    fabric_h, fabric_w = fabric_img.shape[:2]
    bbox_x, bbox_y, bbox_w, bbox_h = bbox_xywh

    # Safety check: does the bbox stay inside the person image
    if bbox_x < 0 or bbox_y < 0 or bbox_w <= 0 or bbox_h <= 0:
        logger.error("BBox coordinates are not valid.")
        return

    if (bbox_x + bbox_w) > person_w or (bbox_y + bbox_h) > person_h:
        logger.error("BBox goes outside the person image.")
        return

    # Step 1: compute the uniform scale
    scale = compute_uniform_scale(fabric_w, fabric_h, bbox_w, bbox_h)

    # Step 2: resize
    resized_fabric = resize_fabric_uniform(fabric_img, scale)

    # Step 3: crop to the exact bbox size
    cropped_fabric = center_crop_to_bbox(resized_fabric, bbox_w, bbox_h)

    # Step 4: create the black canvas
    canvas = create_black_canvas(person_h, person_w, channels)

    # Step 5: paste the cropped fabric at the bbox location
    final_output = paste_fabric_on_canvas(canvas, cropped_fabric, bbox_x, bbox_y)

    # Step 6: save
    # This debug save still uses a fixed disk path (not returned to the
    # caller as a path) -- it is only for manual inspection while
    # testing, it does not affect what fit_fabric_to_bbox() returns.
    os.makedirs(output_folder_path, exist_ok=True)
    output_path = os.path.join(output_folder_path, "fabric_fitted_to_bbox4.png")
    cv2.imwrite(output_path, final_output)

    logger.info("Fitted fabric saved here: %s", output_path)

    return final_output