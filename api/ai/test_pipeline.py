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

PERSON_IMAGE = f"{BASE_DIR}/test_images/ladies12.jpg"

FABRIC_IMAGE = f"{BASE_DIR}/fabric_images/test_fabrics/kurti_fabric6.jpg"

OUTPUT_IMAGE = f"{BASE_DIR}/test_images/ladies_output_img/kurti_17.jpg"

# ----------------------------------------------------------
# Garment Type
#
# "shirt" -> मूळ SAM shirt mask जसाच्या तसा वापरला जातो.
# "kurta" -> shirt mask वरून rule-based Kurta mask तयार होतो.
# ----------------------------------------------------------

GARMENT_TYPE = "shirt"

# ----------------------------------------------------------
# Detection Target
#
# GroundingDINO la konta object DETECT karायcha te sangणara
# text prompt. "shirt", "pant", "jacket" etc. -- garment_type
# peksha vegळa control aahe (template vs detection).
# ----------------------------------------------------------

DETECTION_TARGET = "shirt"

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
    garment_type=GARMENT_TYPE,
    detection_target=DETECTION_TARGET
)

print("\nTesting Completed Successfully.")