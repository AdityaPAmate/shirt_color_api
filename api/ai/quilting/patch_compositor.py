"""
Patch Composition
"""


class PatchCompositor:

    def __init__(self):
        pass

    ####################################################################
    # COMPOSE
    ####################################################################

    def compose(
        self,
        canvas,
        patch,
        mask,
        x,
        y
    ):
        """
        Blend the selected patch into the output canvas.
        """
        raise NotImplementedError