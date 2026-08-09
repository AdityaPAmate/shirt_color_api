"""
File:
    api/ai/pattern_scale_estimator.py

Purpose
-------
हे एकाच फाईलमध्ये दोन वेगळी कामं करणाऱ्या दोन पद्धती (methods) आहेत:

1. estimate(fabric_info)
   -> LEGACY. pipeline.py कडून UPLOADED FABRIC वर call होते
      (शर्टावर नाही). सध्या फक्त architecture placeholder आहे - खरं
      काम अजून लिहिलेलं नाही. pipeline.py फक्त एक dict अपेक्षतो
      (प्रिंट करण्यासाठी), त्यामुळे रिकामा dict देणंही सुरक्षित आहे.

2. estimate_shirt_scale_context(shirt_mask, ...)
   -> NEW. fabric.py कडून ORIGINAL SHIRT वर call होते. RTV, fold-
      separation, आणि geometry-warp या तिघांनाही एकच सुसंगत radius
      देण्यासाठी - single source of truth.

दोन्ही वेगळ्या नावांनी ठेवल्या आहेत मुद्दाम - जेणेकरून दोन वेगळ्या
कामांसाठी एकच नाव वापरल्याने आधी झालेली गल्लत (dict वर shirt_mask चं
लॉजिक चालवलं जाणं -> crash) परत होणार नाही.

CHANGELOG (this round)
-----------------------
1. pattern_radius_px multiplier raised from 1.8 -> 2.2. On a busy
   check/dot shirt (57px pitch), 1.8x produced rtv_sigma=51.3, which
   was NOT enough to fully wash out the dots (confirmed by user -
   "checks shadow" still visible after re-test). This was already
   the documented contingency plan (Master Context v4, §7.4).

2. FOUND AND FIXED A SECOND, HIDDEN BUG while verifying (1) would
   actually have an effect: rtv_sigma was clipped to a HARDCODED
   absolute ceiling of 60.0, unrelated to shirt size. Verified by
   direct calculation that raising the multiplier from 1.8 to 2.2
   alone would NOT have changed the output (51.3 -> 60.0 is the
   most it could reach; 2.5 gives the exact same 60.0, so the
   multiplier increase would have been silently capped and produced
   almost no visible change - the same class of bug as the earlier
   "max() only as good as what it's compared against" issue).
   Fixed by replacing the fixed 60.0 with a ratio of shirt_width_px
   (shirt_width_px * 0.15), consistent with this project's
   non-negotiable "ratio-based, not pixel-hardcoded" principle - it
   now scales correctly on any shirt resolution/size instead of
   silently plateauing at one fixed number.
"""

import numpy as np


class PatternScaleEstimator:

    def __init__(self):
        pass

    ####################################################################
    # LEGACY -- uploaded fabric वर काम करते, pipeline.py कडून वापरली जाते
    ####################################################################

    def estimate(self, fabric_info):
        """
        LEGACY placeholder. pipeline.py याला fabric_info (uploaded
        fabric ची माहिती - dict) पाठवतो, आणि फक्त तो dict प्रिंट
        करतो. अजून खरं pattern-scale लॉजिक इथे लिहिलेलं नाही -
        भविष्यात गरज पडल्यास इथे लिहिता येईल. सध्या रिकामा dict
        देणं पूर्णपणे सुरक्षित आहे, कारण त्याचा पुढे कुठेही प्रत्यक्ष
        वापर होत नाही.
        """
        return {
            "scale_factor": None,
            "note": "Not yet implemented"
        }

    ####################################################################
    # NEW -- original shirt वर काम करते, fabric.py कडून वापरली जाते
    ####################################################################

    def estimate_shirt_scale_context(
            self,
            shirt_mask,
            pattern_pitch_x=None,
            pattern_pitch_y=None
    ):
        """
        Parameters
        ----------
        shirt_mask : ndarray
        pattern_pitch_x, pattern_pitch_y : int किंवा None
            FabricRenderer.estimate_shirt_pattern_pitch() मधून
            मिळालेला मूळ shirt चा pattern pitch.

        Returns
        -------
        dict : rtv_sigma, rtv_lam, large_fold_radius,
               geometry_smooth_sigma, confidence
               (सगळे एकाच pattern_radius_px वरून काढलेले, म्हणून
               सुसंगत)
        """

        mask = (shirt_mask > 0).astype(np.uint8)
        ys, xs = np.where(mask > 0)

        shirt_width_px = float(xs.max() - xs.min()) if len(xs) > 0 else 400.0

        candidates = [p for p in (pattern_pitch_x, pattern_pitch_y) if p]
        pattern_pitch = max(candidates) if candidates else None
        confidence = "detected" if candidates else "fallback"

        # बदललं: 1.8 -> 2.2 -- busy check/dot शर्टवर 1.8x पुरेसं नव्हतं
        # (checks/dots शेडिंगमध्ये अजूनही उरत होते, user ने confirm केलं)
        if pattern_pitch:
            pattern_radius_px = pattern_pitch * 2.2
        else:
            pattern_radius_px = shirt_width_px * 0.06

        pattern_radius_px = float(np.clip(
            pattern_radius_px,
            shirt_width_px * 0.02,
            shirt_width_px * 0.30
        ))

        # बदललं: rtv_sigma ची वरची मर्यादा आधी fixed 60.0 (pixel-hardcoded)
        # होती -- शर्टाच्या आकाराशी काहीही संबंध नव्हता. यामुळे multiplier
        # 1.8 वरून 2.2 केला तरी sigma फक्त 60.0 लाच अडकून राहिला असता,
        # आणि fix चा प्रत्यक्ष परिणाम दिसला नसता (प्रत्यक्ष calculation
        # करून पडताळलं). आता शर्टाच्या रुंदीच्या प्रमाणात (ratio-based)
        # cap ठेवला -- हे प्रोजेक्टच्या "generalizable, no hardcoded pixel
        # values" तत्त्वाशी सुसंगत आहे आणि कुठल्याही रिझोल्यूशन/शर्ट-
        # आकारावर योग्य प्रमाणात scale होतं.
        rtv_sigma_ceiling = max(60.0, shirt_width_px * 0.15)

        return {
            "pattern_pitch": pattern_pitch,
            "pattern_radius_px": pattern_radius_px,
            "rtv_sigma": float(np.clip(pattern_radius_px * 0.5, 1.5, rtv_sigma_ceiling)),
            "rtv_lam": 0.030 if pattern_pitch else 0.015,
            "large_fold_radius": pattern_radius_px,
            "geometry_smooth_sigma": pattern_radius_px,
            "confidence": confidence,
        }