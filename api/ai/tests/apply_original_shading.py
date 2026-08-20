# apply_shirt_shading_relative.py
#
# Purpose:
# Extract RELATIVE shading from the original shirt and apply it
# to the fabric-applied image without transferring the original
# shirt's overall darkness/color.
#
# Key principle:
# We do NOT use absolute original-shirt brightness directly.
#
# Instead:
#   relative_shading = local_illumination / reference_illumination
#
# Therefore approximately:
#   normal area   -> 1.0
#   shadow area   -> < 1.0
#   bright area   -> > 1.0
#
# Final:
#   new_fabric * relative_shading
#
# This helps preserve the original color of the NEW fabric.


from pathlib import Path
import cv2
import numpy as np


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[3]

print("BASE_DIR:", BASE_DIR)


ORIGINAL_IMAGE = BASE_DIR / "test_images" / "person30.png"

FABRIC_APPLIED_IMAGE = (
    BASE_DIR / "test_images" / "shirt_kurta_116.jpg"
)

SHIRT_MASK = (
    BASE_DIR
    / "test_images"
    / "debug"
    / "debug_0.2_shirt_mask.png"
)

OUTPUT_IMAGE = (
    BASE_DIR
    / "test_images"
    / "shirt_kurta_116_with_relative_shading.jpg"
)


# ==========================================================
# PARAMETERS
# ==========================================================

# Large smoothing removes most small fabric patterns/checks
# and keeps broad illumination/shadow.
COARSE_SIGMA = 35

# Medium smoothing keeps more fold information.
MEDIUM_SIGMA = 12

# Broad illumination gets higher importance.
COARSE_WEIGHT = 0.75
MEDIUM_WEIGHT = 0.25

# Controls how strongly the extracted shading changes
# the new fabric.
#
# 1.0 = full extracted relative shading
# 0.0 = no shading; new fabric remains unchanged
SHADING_STRENGTH = 0.70

# Safety limits for relative shading.
MIN_SHADING = 0.80
MAX_SHADING = 1.20

EPSILON = 1e-6


# ==========================================================
# READ IMAGE
# ==========================================================

def read_image(path):
    """Read BGR image and verify it loaded correctly."""
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image:\n{path}"
        )

    return image


# ==========================================================
# READ MASK
# ==========================================================

def read_mask(path, target_size):
    """
    Read garment mask.

    White = garment
    Black = outside garment

    Returns float mask:
        1.0 = garment
        0.0 = outside
    """
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(
            f"Could not read mask:\n{path}"
        )

    mask = cv2.resize(
        mask,
        target_size,
        interpolation=cv2.INTER_NEAREST
    )

    return (mask > 127).astype(np.float32)


# ==========================================================
# GET LUMINANCE
# ==========================================================

def get_luminance(image_bgr):
    """
    Convert BGR image to luminance in range approximately 0-1.

    Shadows and illumination are represented mainly by
    brightness variation, so we work on luminance.
    """
    image = image_bgr.astype(np.float32) / 255.0

    b, g, r = cv2.split(image)

    return (
        0.114 * b +
        0.587 * g +
        0.299 * r
    )


# ==========================================================
# MASKED GAUSSIAN SMOOTHING
# ==========================================================

def masked_gaussian_blur(
    image,
    mask,
    sigma
):
    """
    Gaussian smoothing ONLY using valid shirt pixels.

    A normal Gaussian blur on a masked image would mix black
    pixels from outside the shirt into the shirt boundary.

    To avoid that, use normalized convolution:

        blur(image * mask) / blur(mask)

    This gives a smooth illumination estimate based only on
    pixels belonging to the garment.
    """

    numerator = cv2.GaussianBlur(
        image * mask,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    denominator = cv2.GaussianBlur(
        mask,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    return numerator / (denominator + EPSILON)


# ==========================================================
# EXTRACT RELATIVE SHADING
# ==========================================================

def extract_relative_shading(
    original_bgr,
    shirt_mask
):
    """
    Extract a RELATIVE shading map.

    Important difference from the previous approach:

    OLD:
        absolute brightness of original shirt
            ->
        multiplied with new fabric
            ->
        new fabric can become globally dark

    NEW:
        smooth shirt illumination
            /
        shirt reference illumination
            ->
        relative shading around 1.0

    Example:
        original dark shirt average brightness = 0.30

        normal smooth illumination = 0.30
        0.30 / 0.30 = 1.0

    Therefore the new fabric is NOT globally darkened.

    Only relative variation remains:
        0.90 = slightly darker/shadow
        1.00 = unchanged
        1.08 = slightly brighter
    """

    luminance = get_luminance(original_bgr)

    valid = shirt_mask > 0.5

    if np.count_nonzero(valid) == 0:
        raise ValueError("Shirt mask contains no garment pixels.")

    # ------------------------------------------------------
    # Coarse scale:
    # suppress fine checks, stripes and texture.
    # ------------------------------------------------------
    coarse = masked_gaussian_blur(
        luminance,
        shirt_mask,
        COARSE_SIGMA
    )

    # ------------------------------------------------------
    # Medium scale:
    # retain more fold-related variation.
    # ------------------------------------------------------
    medium = masked_gaussian_blur(
        luminance,
        shirt_mask,
        MEDIUM_SIGMA
    )

    # ------------------------------------------------------
    # Combine illumination estimates.
    # ------------------------------------------------------
    illumination = (
        COARSE_WEIGHT * coarse +
        MEDIUM_WEIGHT * medium
    )

    # ------------------------------------------------------
    # IMPORTANT:
    # Calculate the reference ONLY from shirt pixels.
    #
    # Median is used instead of mean because it is less
    # affected by a few strong shadows or highlights.
    # ------------------------------------------------------
    reference_illumination = np.median(
        illumination[valid]
    )

    # ------------------------------------------------------
    # Convert absolute illumination into RELATIVE shading.
    #
    # Neutral/reference shirt area becomes approximately 1.0.
    # ------------------------------------------------------
    relative_shading = (
        illumination /
        (reference_illumination + EPSILON)
    )

    # ------------------------------------------------------
    # Reduce shading strength around 1.0.
    #
    # Formula:
    #   final = 1 + strength * (relative - 1)
    #
    # If strength = 0:
    #   all pixels become 1.0
    #   -> fabric color unchanged
    #
    # If strength = 1:
    #   full relative shading is used
    # ------------------------------------------------------
    relative_shading = (
        1.0 +
        SHADING_STRENGTH *
        (relative_shading - 1.0)
    )

    # ------------------------------------------------------
    # Limit extreme values.
    # ------------------------------------------------------
    relative_shading = np.clip(
        relative_shading,
        MIN_SHADING,
        MAX_SHADING
    )

    # Outside garment must never modify the image.
    relative_shading[~valid] = 1.0

    return relative_shading


# ==========================================================
# APPLY RELATIVE SHADING
# ==========================================================

def apply_relative_shading(
    fabric_bgr,
    relative_shading,
    shirt_mask
):
    """
    Apply relative shading to the fabric-applied image.

        final = new_fabric * relative_shading

    Since neutral shading is approximately 1.0,
    the new fabric's base color is preserved.
    """

    fabric = fabric_bgr.astype(np.float32) / 255.0

    shading_3ch = np.repeat(
        relative_shading[:, :, np.newaxis],
        3,
        axis=2
    )

    shaded = fabric * shading_3ch

    shaded = np.clip(
        shaded,
        0.0,
        1.0
    )

    mask_3ch = np.repeat(
        shirt_mask[:, :, np.newaxis],
        3,
        axis=2
    )

    # Apply only inside garment.
    result = (
        shaded * mask_3ch +
        fabric * (1.0 - mask_3ch)
    )

    return (result * 255).astype(np.uint8)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\nChecking input files...")

    required_files = [
        ORIGINAL_IMAGE,
        FABRIC_APPLIED_IMAGE,
        SHIRT_MASK
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )

        print("Found:", file_path)

    # ------------------------------------------------------
    # Read images
    # ------------------------------------------------------
    original = read_image(ORIGINAL_IMAGE)
    fabric_applied = read_image(FABRIC_APPLIED_IMAGE)

    # ------------------------------------------------------
    # Match dimensions
    # ------------------------------------------------------
    height, width = original.shape[:2]

    if fabric_applied.shape[:2] != (height, width):
        print(
            "\nResizing fabric-applied image:"
            f"\nFrom: {fabric_applied.shape[:2]}"
            f"\nTo:   {(height, width)}"
        )

        fabric_applied = cv2.resize(
            fabric_applied,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

    # ------------------------------------------------------
    # Read shirt mask
    # ------------------------------------------------------
    shirt_mask = read_mask(
        SHIRT_MASK,
        (width, height)
    )

    # ------------------------------------------------------
    # Extract RELATIVE shading
    # ------------------------------------------------------
    print("\nExtracting relative shading map...")

    relative_shading = extract_relative_shading(
        original,
        shirt_mask
    )

    print(
        "Relative shading range inside shirt:",
        float(np.min(relative_shading[shirt_mask > 0.5])),
        "to",
        float(np.max(relative_shading[shirt_mask > 0.5]))
    )

    # ------------------------------------------------------
    # Apply relative shading
    # ------------------------------------------------------
    print("Applying relative shading to fabric...")

    result = apply_relative_shading(
        fabric_applied,
        relative_shading,
        shirt_mask
    )

    # ------------------------------------------------------
    # Save output
    # ------------------------------------------------------
    OUTPUT_IMAGE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    success = cv2.imwrite(
        str(OUTPUT_IMAGE),
        result
    )

    if not success:
        raise RuntimeError(
            f"Failed to save output:\n{OUTPUT_IMAGE}"
        )

    print("\n" + "=" * 60)
    print("SUCCESS")
    print("=" * 60)
    print("Original Image :", ORIGINAL_IMAGE)
    print("Fabric Image   :", FABRIC_APPLIED_IMAGE)
    print("Shirt Mask     :", SHIRT_MASK)
    print("Output Image   :", OUTPUT_IMAGE)
    print("=" * 60)


if __name__ == "__main__":
    main()