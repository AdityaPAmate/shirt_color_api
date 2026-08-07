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

PERSON_IMAGE = f"{BASE_DIR}/test_images/person17.jpg"

FABRIC_IMAGE = f"{BASE_DIR}/fabric_images/test_fabrics/lines_vertical6.jpg"

OUTPUT_IMAGE = f"{BASE_DIR}/output_tests_fabrics/fabric_output1419.jpg"

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