"""
Patch Extraction
"""


class PatchExtractor:

    def __init__(self):
        pass

    ####################################################################
    # VALIDATE SOURCE
    ####################################################################

    def validate_source(
        self,
        fabric_image,
        patch_size
    ):
        """
        Validate the source texture.
        """
        raise NotImplementedError

    ####################################################################
    # EXTRACT PATCH
    ####################################################################

    def extract_patch(
        self,
        fabric_image,
        x,
        y,
        patch_size
    ):
        """
        Extract one patch.
        """
        raise NotImplementedError

    ####################################################################
    # GET CANDIDATE LOCATIONS
    ####################################################################

    def get_candidate_locations(
        self,
        fabric_image,
        patch_size
    ):
        """
        Return all valid patch locations.
        """
        raise NotImplementedError