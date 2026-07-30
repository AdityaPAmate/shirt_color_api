"""
==========================================================
Pattern Scale Estimator
==========================================================

Purpose
-------
This module converts the analysed fabric information into
a recommended pattern scale.

It DOES NOT modify the fabric.

It DOES NOT generate virtual cloth.

It only decides how large or small the fabric pattern
should appear on the final garment.

Pipeline

Fabric Analyzer
        ↓
Pattern Scale Estimator
        ↓
Virtual Fabric

Future Inputs
-------------

- Pattern Repeat
- Pattern Type
- Pattern Density
- Pattern Direction
- User Preference (optional)

Future Output
-------------

{
    "scale_factor": ...,
    "confidence": ...,
    "method": ...
}
"""


class PatternScaleEstimator:

    def __init__(self):
        pass

    ####################################################################
    # ESTIMATE SCALE
    ####################################################################

    def estimate(self, fabric_info):
        """
        Estimate the pattern scale.

        Current Version
        ---------------

        This milestone does NOT calculate a real scale.

        It only prepares a standard output format that
        future versions will populate.

        Parameters
        ----------
        fabric_info : dict

        Returns
        -------
        dict
        """

        pattern_repeat = fabric_info.get("pattern_repeat")

        return {

            # Future calculated value
            "scale_factor": None,

            # Confidence of estimation
            "confidence": None,

            # Documents how the value was produced
            "method": "pending",

            # Keep original analyzer output
            "pattern_repeat": pattern_repeat
        }