"""
Simple testing file.

This file should remain very small.

All AI logic is handled inside:

    api/ai/pipeline.py
"""

from api.ai.pipeline import ShirtPipeline


from pathlib import Path

# Get the path to the file you want
BASE_DIR = Path(__file__).resolve().parents[2]
print(BASE_DIR)
# ----------------------------------------------------------
# Input Files
# ----------------------------------------------------------

PERSON_IMAGE = f"{BASE_DIR}/test_images/person2.jpg"

FABRIC_IMAGE = f"{BASE_DIR}/fabric_images/plain_green2_fabric.png"

OUTPUT_IMAGE = f"{BASE_DIR}/test_images/fabric_output_q291_fit.jpg"

# ----------------------------------------------------------
# Initialize Pipeline
# ----------------------------------------------------------

pipeline = ShirtPipeline()

# ----------------------------------------------------------
# Run Complete Fabric Replacement Pipeline
# ----------------------------------------------------------

pipeline.replace_fabric(
    person_image_path=PERSON_IMAGE,
    fabric_image_path=FABRIC_IMAGE,
    output_path=OUTPUT_IMAGE,
    fabric_mode="fit"
)

print("\nTesting Completed Successfully.")