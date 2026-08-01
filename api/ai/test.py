from pathlib import Path
import cv2
import numpy as np

# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT = BASE_DIR / "fabric_images" / "plain_green_fabric.jpg"
OUTPUT = BASE_DIR / "test_images" / "fabric_far_view_frequency.png"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)


# ==========================================================
# PARAMETERS FOR FREQUENCY DECOMPOSITION
# ==========================================================
# 1. Base Smoothness (Low Frequency)
#    Controls how much macro-shading/color is preserved.
BASE_BLUR_SIGMA = 3.0

# 2. Thread Weave Attenuation (High Frequency)
#    - 1.00 = Original extreme close-up thread contrast.
#    - 0.00 = Completely flat (paint look).
#    - 0.18 to 0.25 = Distant view where real threads are subtle but visible.
WEAVE_CONTRAST_SCALE = 0.22


def generate_true_fabric_far_view(input_path, output_path):
    print(f"Loading: {input_path}")
    img = cv2.imread(str(input_path))

    if img is None:
        raise FileNotFoundError(f"Image not found at {input_path}")

    # Convert to float32 [0, 1] range for precise linear frequency manipulation
    img_float = img.astype(np.float32) / 255.0

    # ------------------------------------------------------
    # STEP 1: Extract Low-Frequency Base (Broad Color & Shading)
    # ------------------------------------------------------
    # Using a soft Gaussian filter to capture macroscopic tone
    low_freq = cv2.GaussianBlur(
        img_float, (0, 0), sigmaX=BASE_BLUR_SIGMA, sigmaY=BASE_BLUR_SIGMA
    )

    # ------------------------------------------------------
    # STEP 2: Extract High-Frequency Structure (Actual Thread Weave)
    # High_Freq = Original_Image - Low_Freq
    # Contains ONLY the micro-threads, tiny gaps, and real surface texture
    # ------------------------------------------------------
    high_freq = img_float - low_freq

    # ------------------------------------------------------
    # STEP 3: Reconstruct Image with Attenuated Weave Contrast
    # We suppress the thread magnitude to simulate spatial distance,
    # without destroying the structural geometry of the weave.
    # ------------------------------------------------------
    reconstructed = low_freq + (high_freq * WEAVE_CONTRAST_SCALE)

    # ------------------------------------------------------
    # STEP 4: Convert back to uint8 [0, 255] and Save
    # ------------------------------------------------------
    final_output = np.clip(reconstructed * 255.0, 0, 255).astype(np.uint8)

    cv2.imwrite(
        str(output_path), final_output, [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    print(f"Frequency-filtered fabric image saved to: {output_path}")


if __name__ == "__main__":
    generate_true_fabric_far_view(INPUT, OUTPUT)