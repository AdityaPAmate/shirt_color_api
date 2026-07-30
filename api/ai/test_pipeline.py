"""
Simple testing file.

This file should remain very small.

All AI logic is handled inside:

    api/ai/pipeline.py
"""

from api.ai.pipeline import ShirtPipeline
from api.ai.quilting.patch_extractor import PatchExtractor


from pathlib import Path

# Get the path to the file you want
BASE_DIR = Path(__file__).resolve().parents[2]
print(BASE_DIR)
# ----------------------------------------------------------
# Input Files
# ----------------------------------------------------------

PERSON_IMAGE = f"{BASE_DIR}/test_images/person7.jpeg"

FABRIC_IMAGE = f"{BASE_DIR}/fabric_images/design_fabric.jpg"

OUTPUT_IMAGE = f"{BASE_DIR}/test_images/fabric_output_p723_fit.jpg"

# ----------------------------------------------------------
# Initialize Pipeline
# ----------------------------------------------------------

pipeline = ShirtPipeline()

# ----------------------------------------------------------
# Run Complete Fabric Replacement Pipeline
# ----------------------------------------------------------

extractor = PatchExtractor()

locations = extractor.get_candidate_locations(
    FABRIC_IMAGE,
    patch_size=64
)

print(len(locations))

patch = extractor.extract_patch(
    FABRIC_IMAGE,
    locations[0][0],
    locations[0][1],
    64
)

print(patch.shape)

# pipeline.replace_fabric(
#     person_image_path=PERSON_IMAGE,
#     fabric_image_path=FABRIC_IMAGE,
#     output_path=OUTPUT_IMAGE,
#     fabric_mode="fit"
# )

print("\nTesting Completed Successfully.")