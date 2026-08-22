"""
Simple testing file.
All AI logic is handled inside: api/ai/pipeline.py
"""

from api.ai.pipeline_singleton import get_pipeline   # <-- BADLALA

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
print(BASE_DIR)

# ----------------------------------------------------------
# Input Files
# ----------------------------------------------------------

PERSON_IMAGE = f"{BASE_DIR}/test_images/ladies12.jpg"

FABRIC_IMAGE = f"{BASE_DIR}/fabric_images/test_fabrics/kurti_fabric6.jpg"

OUTPUT_IMAGE = f"{BASE_DIR}/test_images/ladies_output_img/kurti_17.jpg"

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

pipeline = get_pipeline()

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