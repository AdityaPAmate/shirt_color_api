"""
File:
    api/ai/fabric_geometry_warp.py

Purpose
-------
Fold-driven geometric displacement of the fabric pattern.

WHY THIS FILE EXISTS (separate from fabric.py)
------------------------------------------------
fabric.py handles SHADING (brightness only) via RTV. That alone makes
a printed pattern look "pasted flat" - lines/checks never bend or
compress at a fold, only get darker/lighter. This module adds the
missing GEOMETRIC cue: it nudges fabric pixels sideways so the
pattern visibly bends AND compresses/stretches at folds, which is
what makes a fabric read as real cloth instead of a flat overlay.

GENERALIZATION PRINCIPLE (important - read before changing numbers)
---------------------------------------------------------------------
Every tunable value here is derived from measurements of the CURRENT
image (shirt width in pixels, detected pattern pitch) - never a fixed
pixel constant. This is deliberate: a fixed pixel number that looks
right on one photo (one resolution, one fabric type) breaks on a
different resolution or a different fabric's pattern scale (this is
exactly what happened when a fixed smoothing radius, tuned on a
stripe fabric, was reused on a check fabric and mistook the checks'
own periodicity for fold geometry - see CHANGELOG).

CHANGELOG
---------
NEW FILE. Extracted out of an earlier inline version that lived in
fabric.py, and generalized:

1. Smoothing radius for the height-field is now derived from the
   shirt's own detected pattern pitch (estimate_shirt_pattern_pitch,
   already computed in fabric.py for RTV). If the pitch is smaller
   than the smoothing radius, the periodic pattern (checks/stripes)
   gets washed out before displacement is computed from it - so
   checks/stripes no longer get mistaken for "large folds" the way
   they did with a fixed sigma=4.

2. Displacement strength is now a FRACTION of shirt width in pixels,
   not a fixed pixel count - scales correctly across image
   resolutions.

3. Edge tapering via distance transform. Displacement is forced to
   zero right at the shirt mask boundary and ramps up smoothly moving
   inward. This fixes an earlier bug where large displacement vectors
   near the collar/shoulder edge pulled in pixels from OUTSIDE the
   shirt mask (skin/background), which looked like "the original
   photo showing through".

NEW (this round): _limit_local_compression() + its two call sites in
   compute_displacement_field(). Root cause investigated and fixed:
   thin printed lines were visually "merging" into a thick blurry
   band wherever the displacement field changed sharply between
   neighbouring pixels (heavy local compression -> cv2.remap's
   linear interpolation blends several source lines into one output
   pixel). This is a RATIO-based clamp (fraction of local spacing
   preserved), not a fixed pixel number, so it self-adjusts to any
   pattern pitch / resolution / fabric - not tuned to one photo.
"""

import cv2
import numpy as np


class FabricGeometryWarp:
    """
    Computes and applies a fold-driven pixel-displacement field to
    the fabric, so the printed pattern visibly bends/compresses at
    real folds instead of staying geometrically flat.
    """

    def __init__(self):
        pass

    ####################################################################
    # LIMIT LOCAL COMPRESSION
    #
    # Prevents the displacement field from compressing neighbouring
    # pixels together so much that they collapse onto (or near) the
    # same output pixel - this is what makes lines look merged /
    # thicker after warping. It is NOT the lines becoming physically
    # wider; it's several thin lines being interpolated into one
    # blurry band by cv2.remap.
    ####################################################################

    def _limit_local_compression(self, field, axis, min_ratio=0.55):
        """
        The LOCAL spacing ratio between two neighbouring pixels after
        warping is approximately (1 + d(field)/d(axis)). A ratio of
        1.0 means "no compression here". A ratio of 0.0 means "these
        two pixels now land on the exact same output pixel" - this is
        where merging happens.

        We clip the gradient of `field` so this ratio can never drop
        below `min_ratio` (e.g. 0.55 -> pixels can be pushed up to
        ~45% closer together - enough to visually read as a fold, but
        never closer than that). This is a RATIO, not a fixed pixel
        count, so it self-adjusts to any pattern pitch, any image
        resolution, any fabric - no per-image tuning needed.

        We then reconstruct the field from the clipped gradient via
        cumulative sum, and re-anchor it to the original field's mean
        (along that axis) so the overall fold shape/position is
        preserved - only the sharp, over-compressing transitions are
        softened.

        Parameters
        ----------
        field : ndarray (float32), the dx or dy displacement map.
        axis : int, 0 for vertical (dy) checks, 1 for horizontal (dx)
               checks.
        min_ratio : float in (0, 1).

        Returns
        -------
        ndarray (float32), same shape as `field`.
        """
        grad = np.gradient(field, axis=axis)

        # स्ट्रेचिंगला (positive gradient) मर्यादा नाही, फक्त कॉम्प्रेशनला
        # (negative gradient) आहे -- कारण फक्त कॉम्प्रेशनमुळेच रेषा
        # एकमेकांत मिसळतात, स्ट्रेचिंगमुळे नाही.
        min_grad = -(1.0 - min_ratio)
        grad_clipped = np.clip(grad, min_grad, None)

        corrected = np.cumsum(grad_clipped, axis=axis).astype(np.float32)

        # मूळ field शी mean जुळवा -> एकूण दुमडण्याची जागा/आकार तोच राहतो,
        # फक्त टोकाचं कॉम्प्रेशन सौम्य होतं
        orig_mean = field.mean(axis=axis, keepdims=True)
        corrected_mean = corrected.mean(axis=axis, keepdims=True)
        corrected = corrected + (orig_mean - corrected_mean)

        return corrected.astype(np.float32)

    ####################################################################
    # COMPUTE DISPLACEMENT FIELD
    ####################################################################

    def compute_displacement_field(
            self,
            shading_map_for_geometry,
            shirt_mask,
            pattern_pitch=None,
            strength_frac=0.02,
            edge_taper_frac=0.05,
            min_smooth_sigma=3.0,
            min_compression_ratio=0.55
    ):
        """
        Parameters
        ----------
        shading_map_for_geometry : ndarray (float, same size as person_image)
            The LARGE-SCALE fold band - i.e. the output of
            separate_real_folds_from_texture(), captured BEFORE
            remove_straight_lines_and_blobs()/enhance_fold_contrast()
            sharpen it. Using the sharpened map here would make the
            warp jittery instead of a smooth fold bend.

        shirt_mask : ndarray
            Binary SAM mask, same H×W as person_image.

        pattern_pitch : int or None
            The ORIGINAL shirt's own detected pattern pitch in pixels
            (from FabricRenderer.estimate_shirt_pattern_pitch() -
            already computed once per render() call for RTV; pass
            the same value here, don't recompute).
            None -> plain shirt / no periodicity detected, a
            size-relative fallback is used instead.

        strength_frac : float
            Max pixel displacement as a FRACTION of shirt width.
            e.g. 0.02 on a 500px-wide shirt -> up to 10px shift.
            Same fraction gives comparable visual strength regardless
            of photo resolution.

        edge_taper_frac : float
            How far inward from the shirt mask boundary (as a
            fraction of shirt width) the displacement ramps from 0
            to full strength. Prevents edge/collar artifacts.

        min_smooth_sigma : float
            Absolute floor for the geometry-smoothing radius, so
            tiny/noisy pitch estimates never produce a near-zero
            (i.e. no-op) smoothing pass.

        min_compression_ratio : float in (0, 1)
            Passed to _limit_local_compression(). Lower -> allows
            stronger fold-squeeze but risks line merging again.
            Higher -> safer against merging but a subtler fold look.

        Returns
        -------
        dx, dy : ndarray, ndarray (float32, same H×W)
        """

        mask = (shirt_mask > 0).astype(np.uint8)
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            h, w = shading_map_for_geometry.shape[:2]
            return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)

        shirt_width_px = float(xs.max() - xs.min())

        # ------------------------------------------------------------
        # STEP 1: Smoothing radius - must exceed the fabric's own
        # periodic pitch, so periodic pattern (checks/stripes) is
        # washed out and only genuine large-scale fold shape remains.
        # ------------------------------------------------------------
        if pattern_pitch and pattern_pitch > 0:
            smooth_sigma = max(pattern_pitch * 1.8, min_smooth_sigma)
        else:
            smooth_sigma = max(shirt_width_px * 0.06, min_smooth_sigma)

        smooth_sigma = min(smooth_sigma, shirt_width_px * 0.25)  # sanity cap

        # ------------------------------------------------------------
        # STEP 2: Treat the smoothed shading map as a height-field,
        # take its gradient -> displacement direction.
        # ------------------------------------------------------------
        h_field = cv2.GaussianBlur(
            shading_map_for_geometry, (0, 0), smooth_sigma
        )

        grad_x = cv2.Sobel(h_field, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(h_field, cv2.CV_32F, 0, 1, ksize=3)

        gx_max = np.abs(grad_x).max()
        gy_max = np.abs(grad_y).max()

        # Flat map (no fold signal at all) -> no displacement, not
        # a divide-by-near-zero blow-up.
        if gx_max < 1e-6 and gy_max < 1e-6:
            h, w = shading_map_for_geometry.shape[:2]
            return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)

        grad_x_norm = grad_x / (gx_max + 1e-6)
        grad_y_norm = grad_y / (gy_max + 1e-6)

        # ------------------------------------------------------------
        # STEP 3: Strength scaled to shirt size, not a fixed pixel
        # count.
        # ------------------------------------------------------------
        strength_px = shirt_width_px * strength_frac

        dx = (-grad_x_norm * strength_px).astype(np.float32)
        dy = (-grad_y_norm * strength_px).astype(np.float32)

        # ------------------------------------------------------------
        # STEP 4 (NEW): कॉम्प्रेशन क्लॅम्प -- taper लावण्याआधी, कारण
        # taper मुळे edge जवळ gradient आधीच बदलतो; आपल्याला "शुद्ध"
        # fold-driven displacement वरच क्लॅम्प लावायचा आहे.
        # dx वर axis=1 (आडवं/x-दिशेतलं कॉम्प्रेशन) तपासतो -- उभ्या
        # रेषा एकमेकांत मिसळणं यासाठी हेच जबाबदार आहे.
        # dy वर axis=0 (उभं/y-दिशेतलं कॉम्प्रेशन) तपासतो -- आडव्या
        # रेषा/पट्ट्यांसाठी.
        # ------------------------------------------------------------
        dx = self._limit_local_compression(dx, axis=1, min_ratio=min_compression_ratio)
        dy = self._limit_local_compression(dy, axis=0, min_ratio=min_compression_ratio)

        # ------------------------------------------------------------
        # STEP 5: Edge taper - displacement must be 0 right at the
        # mask boundary, ramping to full strength moving inward.
        # ------------------------------------------------------------
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        taper_distance_px = max(shirt_width_px * edge_taper_frac, 1.0)
        taper = np.clip(dist / taper_distance_px, 0.0, 1.0)

        dx = dx * taper
        dy = dy * taper

        return dx.astype(np.float32), dy.astype(np.float32)

    ####################################################################
    # APPLY DISPLACEMENT
    ####################################################################

    def apply(self, fabric_image, dx, dy):
        """
        Warps fabric_image pixel coordinates by (dx, dy) using
        cv2.remap. Border pixels reflect instead of going black,
        since the taper (above) already guarantees near-zero
        displacement at the mask edge anyway.
        """
        h, w = fabric_image.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

        map_x = (grid_x + dx).astype(np.float32)
        map_y = (grid_y + dy).astype(np.float32)

        warped = cv2.remap(
            fabric_image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        return warped