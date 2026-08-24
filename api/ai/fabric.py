"""
File:
    api/ai/fabric.py

Purpose:
    This module prepares a fabric image for shirt replacement and
    renders it realistically onto the detected shirt region.

Current Milestone:
    ----------------
    ✔ Period-aligned seamless tiling of the uploaded fabric
    ✔ RTV-based structure/fold extraction (texture-leak free)
    ✔ Lab L-channel-only, highlight-safe shading application
    ✔ Button detection + synthetic button rendering
    ✔ Pocket outline detection (strict, single-candidate)

CHANGELOG (most recent first)
------------------------------
NEW (v13, Cloud-deployment prep track): all Marathi comments/docstrings
     in this file converted to plain English; all print() debug
     statements converted to logger.info() (module-level logger).
     Dead-code cleanup done together with this pass: removed
     validate_inputs(), resize_fabric(), preserve_lighting(),
     extract_fold_map(), clean_fold_map(), apply_fold_map() -- none of
     these were called anywhere in render(), not even as a
     commented-out call. Kept prepare_fabric() (VirtualFabric tiling,
     still reachable via a commented-out call at the top of render()),
     and kept detect_buttons()/draw_synthetic_button(),
     detect_pocket_outline()/draw_pocket_outline(),
     draw_placket_line(), draw_shoulder_seam() -- all still reachable
     via commented-out call sites in render(), left completely
     untouched. Also removed a stale comment block in render() that
     was describing Issue #14's Round 1 logic (blanket coverage-based
     scale-down) -- that logic was already fully replaced by Round 3's
     connected-component filtering and Round 4's morphological-closing
     fix, and an accurate explanation of that already exists right
     below it.
NEW: render() now calls self.prepare_fabric() again (period-aligned
     tiling via VirtualFabric), instead of using the raw fabric
     image directly. Also passes pattern_repeat_y.
NEW: preserve_lighting()'s multiply is NO LONGER applied in render()
     -> it was double-brightening the fabric together with the RTV
     shading map (illumination applied twice, compounding and
     clipping to white on light fabrics). preserve_lighting() itself
     has since been removed as dead code (see v13 note above).
NEW: apply_structure_map_lab() uses a highlight-safe (screen-style)
     blend for shading >= 1.0, so already-light fabrics do not blow
     out to pure white and lose colour/pattern.
NEW: separate_real_folds_from_texture() splits the RTV shading map
     into a large-scale (real fold) band and a fine-scale (residual
     texture/print-leak) band using a bilateral filter, and only
     suppresses the fine band based on busyness.
NEW: enhance_fold_contrast() uses unsharp-masking (edge-aware) gain
     instead of a flat multiplier, so fold edges look crisp instead
     of blobby/artificial.
NEW: estimate_shirt_busyness() scores how "busy" (print-heavy) the
     original shirt is, used to scale RTV smoothing + suppression.
NEW: detect_buttons() / draw_synthetic_button() -> classical Hough
     Circle detection along the placket + procedurally rendered
     buttons (with non-max suppression + capped count).
NEW: detect_pocket_outline() / draw_pocket_outline() -> strict,
     single-best-candidate contour detection for the pocket outline.
"""

import logging

import cv2
import numpy as np
from api.ai.virtual_fabric import VirtualFabric
from pathlib import Path
from api.ai.rtv_smoothing import extract_rtv_structure
from api.ai.fabric_downsampling import fit_fabric_to_bbox
from api.ai.fabric_analyzer import FabricAnalyzer
from api.ai.utils import log_execution_time

logger = logging.getLogger(__name__)


class FabricRenderer:
    """
    FabricRenderer is responsible for preparing the uploaded fabric
    and rendering it realistically onto the detected shirt region.

    This class DOES NOT:
        - detect the shirt
        - generate the SAM mask
        - save images (except debug snapshots)
    """

    def __init__(self):
        """
        Initialize helper classes.
        """
        self.virtual_fabric = VirtualFabric()
        self.fabric_analyzer = FabricAnalyzer()

    ####################################################################
    # PREPARE FABRIC
    ####################################################################

    def prepare_fabric(
            self,
            fabric_image,
            target_width,
            target_height,
            repeat_size=None,
            repeat_size_y=None
    ):
        """
        Prepare the fabric before rendering.

        Delegates to VirtualFabric.generate(), which performs
        period-aligned seamless tiling (see virtual_fabric.py).

        Not currently called from render() -- the active path is
        fit_fabric_to_bbox() instead. Kept here, reachable only via a
        commented-out call at the top of render(), in case the
        period-aligned tiling approach needs to be switched back to.
        """

        return self.virtual_fabric.generate(
            fabric_image=fabric_image,
            target_width=target_width,
            target_height=target_height,
            repeat_size=repeat_size,
            repeat_size_y=repeat_size_y
        )

    @log_execution_time
    def estimate_original_shirt_pattern_repeat(self, person_image, shirt_mask):
        mask_bin = (shirt_mask > 0).astype(np.uint8)
        ys, xs = np.where(mask_bin > 0)
        if len(ys) == 0:
            return None, None

        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1

        crop = person_image[y0:y1, x0:x1].copy()
        crop_mask = mask_bin[y0:y1, x0:x1]

        non_shirt = (crop_mask == 0).astype(np.uint8) * 255
        crop = cv2.inpaint(crop, non_shirt, 9, cv2.INPAINT_TELEA)

        repeat_x = self.fabric_analyzer.detect_pattern_repeat(crop)
        repeat_y = self.fabric_analyzer.detect_pattern_repeat_y(crop)

        # ------------------------------------------------------------
        # If autocorrelation fails (returns None), use the FFT fallback.
        # ------------------------------------------------------------
        if repeat_x is None:
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            repeat_x = self.fabric_analyzer.detect_pattern_repeat_fft_fallback(gray_crop)
            logger.info(
                "Autocorrelation failed, using FFT fallback | repeat_x: %s",
                repeat_x
            )

        return repeat_x, repeat_y

    ####################################################################
    # RTV STRUCTURE EXTRACTION
    ####################################################################
    @log_execution_time
    def extract_structure_map_rtv(
            self,
            person_image,
            shirt_mask,
            pattern_repeat=None,
            busyness=0.0
    ):
        """
        The sigma cap is no longer a fixed 8.0 -- it is now adaptive,
        based on the actual size of the shirt crop. The old fixed cap
        was too small for large checks (e.g. pitch=38px) -- both
        pitch=38 and pitch=420 were being clipped to the same
        sigma=8.0, so Fix #1's output showed no visible difference.
        """

        mask_bin = (shirt_mask > 0).astype(np.uint8)
        ys, xs = np.where(mask_bin > 0)

        if len(ys) > 0:
            crop_h = ys.max() - ys.min()
            crop_w = xs.max() - xs.min()
            crop_min_dim = max(1, min(crop_h, crop_w))
        else:
            crop_min_dim = 200

        # sigma can go up to 18% of the shirt crop size (minimum stays
        # 8.0, so the old behavior is preserved for fine weave textures)
        sigma_cap = max(8.0, crop_min_dim * 0.18)

        sigma = 3.0
        lam = 0.015

        # pitch_sigma is kept separate -- the busyness-boost step below
        # needs to know whether a reliable sigma was already determined
        # from the pitch.
        pitch_sigma = None

        if pattern_repeat:
            try:
                pitch = min(pattern_repeat) if isinstance(pattern_repeat, (tuple, list)) else pattern_repeat
                if pitch and pitch > 0:
                    pitch_sigma = max(1.5, min(sigma_cap, pitch * 0.55))
                    sigma = pitch_sigma
            except Exception:
                pass

        if busyness > 0.5:
            if pitch_sigma is not None:
                # Previously sigma was unconditionally set to
                # sigma_cap*0.5 here -- this fully overrode a precise
                # sigma already set from the pitch (calibrated from the
                # fabric's own repeat spacing), even when one existed.
                # pitch_sigma is a more reliable base, so for busyness
                # we now only apply a mild (30%) boost instead of
                # jumping to sigma_cap*0.5.
                sigma = max(sigma, pitch_sigma * 1.3)
            else:
                # Only use the old aggressive fallback strategy when
                # the pitch is unknown (detection failed).
                sigma = max(sigma, sigma_cap * 0.5)
            lam = 0.03

        # ---- Keep this log line -- use it to confirm the actual sigma
        # on the next run ----
        logger.info(
            "RTV params | pitch: %s | crop_min_dim: %s | sigma_cap: %.2f | "
            "final sigma: %.2f | lam: %s",
            pattern_repeat, crop_min_dim, sigma_cap, sigma, lam
        )

        # Changed iterations from 6 to 4.
        #
        # Every iteration inside rtv_smooth() runs an expensive sparse
        # direct solve (spsolve) -- this is the single biggest time
        # cost in the whole pipeline (confirmed from timer logs:
        # extract_rtv_structure takes 77-109 sec, about 65-85% of the
        # whole render() call).
        #
        # Both the original RTV paper (Xu et al., SIGGRAPH Asia 2012)
        # and rtv_smooth()'s own default suggest 4 iterations -- 6 was
        # extra cost that didn't meaningfully improve quality (even
        # though RTV is an iterative refinement, most of the
        # structure/texture separation already converges within the
        # first few iterations).
        RTV_ITERATIONS = 4

        shading_map = extract_rtv_structure(
            person_image, shirt_mask, sigma=sigma, lam=lam, iterations=RTV_ITERATIONS
        )
        return shading_map

    ####################################################################
    # SHIRT BUSYNESS ESTIMATION
    ####################################################################
    @log_execution_time
    def estimate_shirt_busyness_map(self, person_image, shirt_mask, window=15):
        """
        Instead of a fixed constant (/40.0), this normalizes adaptively
        based on each photo's own actual contrast range. This means
        busyness_map always spreads across the full [0,1] range even
        across photos with different exposure/resolution -- there is
        no need to "calibrate on test images" anymore.
        """
        gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3).astype(np.float32)

        mean = cv2.boxFilter(lap, -1, (window, window))
        mean_sq = cv2.boxFilter(lap * lap, -1, (window, window))
        local_std = np.sqrt(np.clip(mean_sq - mean * mean, 0, None))

        mask = (shirt_mask > 0)

        if mask.sum() == 0:
            return np.zeros_like(local_std, dtype=np.float32)

        # ------------------------------------------------------------
        # Boundary-contamination fix.
        #
        # Pixels near the shirt's edge (next to the background) get an
        # artificially high local_std -- because the 15x15 window mixes
        # the strong contrast of both the shirt AND the background
        # (this is not real print/busyness, just a segmentation-edge
        # artifact).
        #
        # Since ref_high (95th percentile) is a single normalization
        # threshold for the whole map, these slightly contaminated edge
        # pixels can pull that threshold up unnecessarily -> skewing
        # the entire busyness_map.
        #
        # So ref_high is now computed only from the shirt's true
        # interior region (mask eroded by a kernel the size of the
        # window). The final busyness_map still covers the full mask
        # (below) -- only the normalization reference has been cleaned
        # up.
        # ------------------------------------------------------------
        erosion_kernel = np.ones((window, window), np.uint8)
        interior_mask = cv2.erode(mask.astype(np.uint8), erosion_kernel) > 0

        reference_mask = interior_mask if interior_mask.sum() > 0 else mask

        ref_high = np.percentile(local_std[reference_mask], 95)
        ref_high = max(ref_high, 1e-3)  # avoid divide-by-zero

        busyness_map = np.clip(local_std / ref_high, 0.0, 1.0)

        logger.info(
            "Busyness map stats | ref_high (95th pct, interior-only): %.4f | "
            "mean over shirt: %.4f | max: %.4f",
            ref_high,
            float(np.mean(busyness_map[mask])),
            float(np.max(busyness_map[mask]))
        )

        return busyness_map * mask.astype(np.float32)

    ####################################################################
    # CHROMATICITY-BASED MATERIAL EDGE MASK
    # (Classical Intrinsic Image Decomposition -- Retinex-style
    # colour-ratio test. No Neural Network here, just gradients.)
    ####################################################################
    @log_execution_time
    def estimate_material_edge_mask(self, person_image, shirt_mask):
        """
        Region-based classical intrinsic decomposition.

        The previous version only worked on the color-change EDGE
        (a thin boundary), so the inside of a zipper/panel stayed
        untouched (still detailed).

        Now: it finds the DOMINANT (most common) color of the shirt
        region, and measures how far every pixel is from that color.
        Any region that is significantly different from the dominant
        color (zipper, a differently-colored panel, trim) -- gets
        neutralized as a whole region (not just the edge --
        morphological closing fills in the inside too).

        Shading (i.e. real folds) in the parts that are the original
        fabric color is left untouched.
        """

        lab = cv2.cvtColor(person_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        mask_bool = shirt_mask > 0
        if mask_bool.sum() == 0:
            return np.zeros(person_image.shape[:2], dtype=np.float32)

        ys, xs = np.where(mask_bool)
        crop_min_dim = max(1, min(ys.max() - ys.min(), xs.max() - xs.min()))

        # ------------------------------------------------------------
        # Dominant fabric color -- the MEDIAN of a/b over the shirt
        # region (ignores large outlier regions like zippers/trim,
        # because the original fabric region is the majority)
        # ------------------------------------------------------------
        a_dom = float(np.median(a[mask_bool]))
        b_dom = float(np.median(b[mask_bool]))

        chroma_dist = np.sqrt((a - a_dom) ** 2 + (b - b_dom) ** 2)

        # ------------------------------------------------------------
        # Adaptive threshold -- based on the 70th percentile of
        # chroma_dist within the shirt region itself (not fixed)
        # ------------------------------------------------------------
        ref_high = np.percentile(chroma_dist[mask_bool], 70)
        ref_high = max(ref_high, 1e-3)

        region_weight = np.clip(chroma_dist / ref_high, 0.0, 1.0)

        # Squared to exclude minor/subtle color drift (which can also
        # happen just from shading) -- only regions that are genuinely
        # a different color will get a high weight
        region_weight = region_weight ** 2

        # ------------------------------------------------------------
        # Morphological CLOSING -- the previous version only caught the
        # boundary, now the inside of the zipper/panel gets filled in
        # as one solid region too (kernel size is adaptive to the crop
        # size)
        # ------------------------------------------------------------
        kernel_size = max(3, int(crop_min_dim * 0.03))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        region_u8 = (region_weight * 255).astype(np.uint8)
        region_u8 = cv2.morphologyEx(region_u8, cv2.MORPH_CLOSE, kernel)
        region_weight = region_u8.astype(np.float32) / 255.0

        # Soft transition at the edge (to avoid a hard cutoff)
        blur_sigma = max(2.0, min(15.0, crop_min_dim * 0.03))
        region_weight = cv2.GaussianBlur(region_weight, (0, 0), blur_sigma)

        region_weight = region_weight * mask_bool.astype(np.float32)

        logger.info(
            "estimate_material_edge_mask | crop_min_dim: %s | a_dom: %.1f | "
            "b_dom: %.1f | ref_high (70th pct): %.2f | kernel_size: %s | "
            "blur_sigma: %.2f | mean region_weight over shirt: %.4f",
            crop_min_dim, a_dom, b_dom, ref_high, kernel_size, blur_sigma,
            float(np.mean(region_weight[mask_bool]))
        )

        return region_weight

    ####################################################################
    # FLAT-PATCH DETECTOR (achromatic logos/trims -- catches what
    # chroma alone misses)
    ####################################################################
    @log_execution_time
    def estimate_flat_patch_mask(self, person_image, shirt_mask):
        """
        The chroma-based check (a/b) only catches colored material
        changes. But logo/trim regions are often gray-on-gray
        (achromatic) -- a/b stays nearly the same there, so the chroma
        check misses them.

        The difference: a real fold is always GRADUAL (continuously
        changing). A logo/trim/patch, on the other hand, is a flat
        region with a sudden (sharp) boundary. This function finds
        that difference by computing the L (brightness) std over two
        different window sizes.
        """

        lab = cv2.cvtColor(person_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        mask_bool = shirt_mask > 0
        if mask_bool.sum() == 0:
            return np.zeros(person_image.shape[:2], dtype=np.float32)

        ys, xs = np.where(mask_bool)
        crop_min_dim = max(1, min(ys.max() - ys.min(), xs.max() - xs.min()))

        small_win = max(3, int(crop_min_dim * 0.015))
        if small_win % 2 == 0:
            small_win += 1
        medium_win = max(7, int(crop_min_dim * 0.06))
        if medium_win % 2 == 0:
            medium_win += 1

        mean_s = cv2.boxFilter(L, -1, (small_win, small_win))
        mean_sq_s = cv2.boxFilter(L * L, -1, (small_win, small_win))
        std_small = np.sqrt(np.clip(mean_sq_s - mean_s ** 2, 0, None))

        mean_m = cv2.boxFilter(L, -1, (medium_win, medium_win))
        mean_sq_m = cv2.boxFilter(L * L, -1, (medium_win, medium_win))
        std_medium = np.sqrt(np.clip(mean_sq_m - mean_m ** 2, 0, None))

        ref_medium = np.percentile(std_medium[mask_bool], 85)
        ref_medium = max(ref_medium, 1e-3)
        medium_norm = np.clip(std_medium / ref_medium, 0.0, 1.0)

        # real fold: small_std ≈ medium_std (continuous change) ->
        # flatness_ratio ≈ 1
        # flat patch: small_std << medium_std (flat inside, jump at
        # boundary) -> ratio ≈ 0
        flatness_ratio = std_small / (std_medium + 1e-3)
        flatness_ratio = np.clip(flatness_ratio, 0.0, 1.0)

        plateau_score = medium_norm * (1.0 - flatness_ratio)

        kernel_size = max(3, int(crop_min_dim * 0.02))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        plateau_u8 = (plateau_score * 255).astype(np.uint8)
        plateau_u8 = cv2.morphologyEx(plateau_u8, cv2.MORPH_CLOSE, kernel)
        plateau_score = plateau_u8.astype(np.float32) / 255.0

        blur_sigma = max(2.0, min(15.0, crop_min_dim * 0.03))
        plateau_score = cv2.GaussianBlur(plateau_score, (0, 0), blur_sigma)
        plateau_score = plateau_score * mask_bool.astype(np.float32)

        logger.info(
            "estimate_flat_patch_mask | crop_min_dim: %s | small_win: %s | "
            "medium_win: %s | mean plateau_score over shirt: %.4f",
            crop_min_dim, small_win, medium_win,
            float(np.mean(plateau_score[mask_bool]))
        )

        return plateau_score

    ####################################################################
    # RESIDUAL FINE-GRAIN CLEANUP (final polish pass)
    ####################################################################
    @log_execution_time
    def remove_residual_fine_grain(self, shading_map, shirt_mask):
        """
        A final edge-preserving smoothing pass to remove fine
        grain/noise texture that is still left over in regions like
        the sleeve, even after all the steps above. It preserves large
        fold shapes (large gradients) and only removes fine, grainy
        noise.
        """

        mask_bool = shirt_mask > 0
        if mask_bool.sum() == 0:
            return shading_map

        ys, xs = np.where(mask_bool)
        crop_min_dim = max(1, min(ys.max() - ys.min(), xs.max() - xs.min()))

        map_min, map_max = shading_map.min(), shading_map.max()
        norm = (shading_map - map_min) / (map_max - map_min + 1e-6)
        norm_u8 = (norm * 255).astype(np.uint8)

        grain_radius = max(10.0, crop_min_dim * 0.06)
        intensity_std = float(np.std(norm_u8[mask_bool]))
        grain_sigma_color = max(10.0, min(60.0, intensity_std * 0.8))

        smoothed_u8 = cv2.bilateralFilter(
            norm_u8, d=0, sigmaColor=grain_sigma_color, sigmaSpace=grain_radius
        )

        smoothed = (smoothed_u8.astype(np.float32) / 255.0) * \
                   (map_max - map_min) + map_min

        logger.info(
            "remove_residual_fine_grain | crop_min_dim: %s | grain_radius: %.2f | "
            "grain_sigma_color: %.2f",
            crop_min_dim, grain_radius, grain_sigma_color
        )

        return smoothed

    ####################################################################
    # DEBUG VISUALIZATION HELPER (adaptive -- no fixed multiplier)
    ####################################################################
    @log_execution_time
    def _debug_visualize_map(self, value_map):
        """
        shading_map values are typically in the 0.3-1.6 (float) range.
        If this array is passed to cv2.imwrite() as-is, it reads each
        pixel as nearly 0/1 (uint8) -> the whole image looks black.

        This function adaptively stretches the range to 0-255 based on
        each image's own ACTUAL min/max -- so every different photo
        (even with a different shading range) will always show the
        correct contrast. No fixed multiplier (like the old *127) is
        used here.
        """
        v_min = float(value_map.min())
        v_max = float(value_map.max())

        if v_max - v_min < 1e-6:
            return np.full(value_map.shape, 128, dtype=np.uint8)

        normalized = (value_map - v_min) / (v_max - v_min)
        return (normalized * 255).astype(np.uint8)

    ####################################################################
    # SEPARATE REAL FOLDS FROM RESIDUAL TEXTURE
    ####################################################################
    @log_execution_time
    def separate_real_folds_from_texture(
            self, shading_map, busyness_map=None, busyness=0.0,
            large_fold_radius=25, pattern_pitch=None
    ):
        """
        The bilateral radius (sigmaSpace) was already made larger than
        the check/line pitch (Issue #6).

        (Issue #8 fix): sigmaColor is no longer FIXED at 45. In a
        bilateral filter, sigmaSpace (radius) and sigmaColor (intensity
        threshold) are two INDEPENDENT axes -- no matter how large the
        radius is, if a check pattern's intensity contrast is higher
        than the fixed sigmaColor, the filter treats those check edges
        as a "real edge" and leaves them alone (does not blur them) --
        meaning checks keep leaking through even with a fixed radius.

        Now sigmaColor is derived adaptively from the ACTUAL intensity
        spread within the shirt region (the std of norm_u8, masked by
        busyness_map) -- higher sigmaColor for busier/high-contrast
        prints, lower for less busy shirts.
        """

        if pattern_pitch:
            # always keep the radius at least 1.8x the pitch
            large_fold_radius = max(large_fold_radius, int(pattern_pitch * 1.8))

        map_min, map_max = shading_map.min(), shading_map.max()
        norm = (shading_map - map_min) / (map_max - map_min + 1e-6)
        norm_u8 = (norm * 255).astype(np.uint8)

        # ------------------------------------------------------------
        # sigmaColor is adaptive -- based on the actual intensity
        # spread within the shirt region (busyness_map>0 = shirt
        # pixels, everything else is outside)
        # ------------------------------------------------------------
        if busyness_map is not None:
            mask_region = busyness_map > 0
        else:
            mask_region = np.ones_like(norm_u8, dtype=bool)

        if mask_region.sum() > 0:
            intensity_std = float(np.std(norm_u8[mask_region]))
        else:
            intensity_std = float(np.std(norm_u8))

        sigma_color = max(15.0, min(80.0, intensity_std * 1.2))

        logger.info(
            "separate_real_folds | intensity_std: %.2f | adaptive sigmaColor: %.2f",
            intensity_std, sigma_color
        )

        large_scale_u8 = cv2.bilateralFilter(
            norm_u8, d=0, sigmaColor=sigma_color, sigmaSpace=large_fold_radius
        )
        large_scale = (large_scale_u8.astype(np.float32) / 255.0) * \
                      (map_max - map_min) + map_min

        fine_residual = shading_map - large_scale

        if busyness_map is not None:
            bm = cv2.resize(busyness_map, (shading_map.shape[1], shading_map.shape[0]))
            # more aggressive suppression: was 0.9->0.10 before (only
            # ~45% max suppression), now 0.7->0.02 (up to ~98% max
            # suppression)
            fine_blend_map = 0.7 - (0.7 - 0.02) * bm
            fine_blend_map = np.clip(fine_blend_map, 0.02, 0.7)
            fine_residual = fine_residual * fine_blend_map
        else:
            fine_blend = float(np.interp(busyness, [0.0, 1.0], [0.7, 0.02]))
            fine_residual = fine_residual * fine_blend

        logger.info(
            "separate_real_folds | pattern_pitch: %s | large_fold_radius used: %s",
            pattern_pitch, large_fold_radius
        )

        return large_scale + fine_residual

    ####################################################################
    # ENHANCE FOLD CONTRAST (edge-aware unsharp masking)
    ####################################################################
    @log_execution_time
    def enhance_fold_contrast(
            self,
            shading_map,
            shirt_mask=None,
            edge_gain=1.8,
            smooth_sigma=8,
            clip_range=None
    ):
        """
        Unsharp-masking based contrast enhancement:
        only amplifies LOCAL EDGES (real fold transitions), leaves
        flat regions untouched -> looks like a crease, not a blob.

        (Issue #7 fix): clip_range is no longer FIXED at (0.75, 1.30).
        The old fixed clip_range was so narrow that tanh() always
        saturated at the same [0.75, 1.30] bounds -- no matter how
        sigma/large_fold_radius changed upstream, the final output
        always ended up in nearly the same range/std (verified via
        logs: min/max were stuck close to 0.75/1.30 in every run).

        Now clip_range is derived adaptively from the std of the
        ACTUAL enhanced values within the shirt region -- the stronger
        the real fold/texture signal, the more room tanh gets, so that
        upstream fixes actually show up in the final output.

        Uses a soft (tanh) clip instead of a hard clip, and a light
        final blur, to avoid harsh/plastic-looking edges.
        """

        very_smooth = cv2.GaussianBlur(shading_map, (0, 0), smooth_sigma)

        edge_component = shading_map - very_smooth

        enhanced = very_smooth + edge_component * edge_gain

        # ------------------------------------------------------------
        # clip_range is adaptive, based on the actual spread within
        # the shirt region
        # ------------------------------------------------------------
        if clip_range is None:
            if shirt_mask is not None:
                mask_bool = shirt_mask > 0
                sample = enhanced[mask_bool] if mask_bool.sum() > 0 else enhanced
            else:
                sample = enhanced

            measured_std = float(np.std(sample))

            # keep a minimum half_range of 0.15 (to avoid the range
            # going near zero on a very flat/plain shirt and causing a
            # divide-by-near-zero)
            half_range = max(0.15, measured_std * 3.0)

            center = 1.0
            clip_range = (center - half_range, center + half_range)

        lo, hi = clip_range
        center = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0

        # (transparency fix): this used to log the std of the entire
        # (un-masked) enhanced array, while clip_range was actually
        # derived from the std of just the shirt region (masked
        # sample) -- two different numbers that looked like one in the
        # log. To avoid this confusion, this now logs the real masked
        # std (the same one used in the calculation).
        logger.info(
            "enhance_fold_contrast | measured_std(masked): %.5f | clip_range used: (%.3f, %.3f)",
            measured_std, lo, hi
        )

        enhanced = center + half_range * np.tanh(
            (enhanced - center) / half_range
        )

        enhanced = cv2.GaussianBlur(enhanced, (0, 0), 1.2)

        return enhanced

    ####################################################################
    # APPLY STRUCTURE MAP (Lab L-channel, highlight-safe)
    ####################################################################
    @log_execution_time
    def apply_structure_map_lab(
            self,
            fabric_image,
            shading_map
    ):
        """
        Multiplicative shading application on the L channel only
        (Lab space) - fabric colour (a, b channels) is left
        completely untouched, so hue/chroma never shifts.

        Highlight-safe blend: when shading >= 1.0 (brightening),
        a screen-style blend is used instead of a hard multiply,
        so it asymptotically approaches 255 instead of clipping -
        this prevents already-light fabrics from washing out to
        flat white and losing their pattern/colour.
        """

        lab = cv2.cvtColor(fabric_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        brighten_mask = shading_map >= 1.0

        L_out = np.empty_like(L)

        # (Issue #14, Round 3 fix): adaptive instead of a fixed
        # shadow_strength=2.9.
        #
        # The old fixed 2.9x multiplier assumed the deviation in
        # shading_map was always subtle (so it needed a big boost to
        # make real folds stand out). But when shading_map already has
        # a large deviation (e.g. a strong signal preserved by RTV
        # because of embroidery motifs), this same fixed multiplier
        # darkens it way too much -- this shows up clearly, especially
        # on white/light new fabric.
        #
        # Now the strength is derived adaptively from the 95th
        # percentile of the ACTUAL darkening-deviation within the
        # shirt region -- the goal being that even the darkest (95th
        # percentile) pixel gets at most ~40% darker, not more. If the
        # deviation is already large (a strong shading signal), the
        # strength automatically comes down.
        darken_deviation = 1.0 - shading_map[~brighten_mask]

        if darken_deviation.size > 0:
            ref_deviation = float(np.percentile(darken_deviation, 95))
        else:
            ref_deviation = 0.0

        ref_deviation = max(ref_deviation, 1e-3)

        TARGET_MAX_DARKEN = 0.40  # the 95th-percentile pixel should
        # darken by at most this fraction

        shadow_strength = np.clip(TARGET_MAX_DARKEN / ref_deviation, 1.0, 3.0)

        logger.info(
            "apply_structure_map_lab | adaptive shadow_strength: %.3f "
            "(ref_deviation 95th pct: %.4f)",
            shadow_strength, ref_deviation
        )

        # when darkening (shading < 1) -> a simple multiply is enough
        shadow_map = 1.0 - (1.0 - shading_map[~brighten_mask]) * shadow_strength
        shadow_map = np.clip(shadow_map, 0.0, 1.0)

        L_out[~brighten_mask] = L[~brighten_mask] * shadow_map

        # when brightening (shading >= 1) -> screen-style soft blend
        excess = shading_map[brighten_mask] - 1.0
        L_out[brighten_mask] = 255 - (255 - L[brighten_mask]) * (1.0 - excess * 0.2)

        L_out = np.clip(L_out, 0, 255)

        lab[:, :, 0] = L_out

        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        return result

    ####################################################################
    # BUTTONS
    ####################################################################

    def detect_buttons(self, person_image, shirt_mask, max_buttons=8):
        """
        Detects likely button locations along the shirt's center
        placket using Hough Circle detection, restricted to a
        narrow vertical strip, with non-max suppression to remove
        duplicate/overlapping detections.
        """

        gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)
        mask = (shirt_mask > 0).astype(np.uint8)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return []

        x_center = int(np.median(xs))
        strip_half_width = max(10, int((xs.max() - xs.min()) * 0.05))

        strip_mask = np.zeros_like(mask)
        strip_mask[:, max(0, x_center - strip_half_width): x_center + strip_half_width] = 1
        strip_mask = strip_mask & mask

        region = cv2.bitwise_and(gray, gray, mask=strip_mask)
        region_blur = cv2.GaussianBlur(region, (3, 3), 0)

        circles = cv2.HoughCircles(
            region_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=60,
            param2=11,
            minRadius=5,
            maxRadius=11
        )

        if circles is None:
            return []

        raw = [(int(c[0]), int(c[1]), int(c[2])) for c in circles[0]]
        raw = [c for c in raw if strip_mask[c[1], c[0]] > 0]

        # --------------------------------------------------------------
        # Non-max suppression -> avoid multiple circles for the same
        # button
        # --------------------------------------------------------------
        deduped = []
        for (x, y, r) in raw:
            too_close = any(
                np.hypot(x - kx, y - ky) < 25 for (kx, ky, kr) in deduped
            )
            if not too_close:
                deduped.append((x, y, r))

        # --------------------------------------------------------------
        # Limit the maximum number of buttons (Hough already returns
        # them sorted by confidence, so taking the top-N is safe)
        # --------------------------------------------------------------
        deduped = deduped[:max_buttons]

        return deduped

    def draw_synthetic_button(self, image, x, y, r):
        """
        Procedurally draws a small, generic (fabric-colour-based)
        button. Uses only 2 subtle thread holes and low opacity to
        avoid looking like a stark black/white "football" pattern.
        """

        overlay = image.copy()

        patch = image[max(0, y - 2 * r):y + 2 * r, max(0, x - 2 * r):x + 2 * r]
        if patch.size > 0:
            mean_c = patch.reshape(-1, 3).mean(axis=0)
            base_color = tuple(int(min(255, c * 1.15 + 15)) for c in mean_c)
        else:
            base_color = (225, 225, 225)

        cv2.circle(overlay, (x, y), r, base_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x, y), r, (100, 100, 100), 1, lineType=cv2.LINE_AA)

        # just one subtle highlight
        cv2.circle(
            overlay, (x - r // 3, y - r // 3), max(1, r // 4),
            (255, 255, 255), -1, lineType=cv2.LINE_AA
        )

        # just 2 subtle, low-contrast thread holes
        hole_r = max(1, r // 6)
        cv2.circle(overlay, (x - r // 4, y), hole_r, (120, 120, 120), -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x + r // 4, y), hole_r, (120, 120, 120), -1, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.82, image, 0.18, 0, dst=image)
        return image

    ############################################################
    def draw_placket_line(self, image, shirt_mask):
        """
        Draws a light dashed stitch-line down the middle of the shirt,
        near the button row -> makes the shirt look "stitched". Purely
        geometric (mask-based), no AI.
        """
        mask = (shirt_mask > 0).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return image

        x_center = int(np.median(xs))
        y_top, y_bottom = ys.min(), ys.max()

        dash_len = 6
        gap_len = 4
        y = y_top

        while y < y_bottom:
            y_end = min(y + dash_len, y_bottom)
            if mask[y, x_center] > 0 and mask[y_end - 1, x_center] > 0:
                cv2.line(
                    image, (x_center, y), (x_center, y_end),
                    (60, 60, 60), 1, lineType=cv2.LINE_AA
                )
            y += dash_len + gap_len

        return image

    ####################################################################
    # POCKET
    ####################################################################

    def detect_pocket_outline(self, person_image, shirt_mask):
        """
        Detects a single, best-candidate pocket outline using strict
        area / aspect-ratio / shape / fill-ratio filters, so fabric
        texture edge-noise is not mistaken for a pocket.

        Returns at most one contour (a shirt has one chest pocket,
        or none at all). Returning nothing is preferred over a
        false-positive outline.
        """

        gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)
        mask = (shirt_mask > 0).astype(np.uint8) * 255

        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        ys, xs = np.where(mask > 0)
        shirt_h = ys.max() - ys.min()
        shirt_w = xs.max() - xs.min()
        shirt_area = shirt_h * shirt_w

        best_candidate = None
        best_area = 0

        for c in contours:
            area = cv2.contourArea(c)

            # a fairly large minimum-area threshold -> filters out
            # fabric texture noise
            if area < 0.015 * shirt_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h + 1e-6)
            if not (0.7 < aspect < 2.2):
                continue

            rel_y = (y - ys.min()) / float(shirt_h)
            if not (0.18 < rel_y < 0.45):
                continue

            # check whether the shape is roughly rectangular
            approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
            if not (4 <= len(approx) <= 8):
                continue

            # require a solid, filled shape -- not a scattered, broken
            # outline
            fill_ratio = area / float(w * h + 1e-6)
            if fill_ratio < 0.55:
                continue

            # keep only the largest, most reliable candidate
            if area > best_area:
                best_area = area
                best_candidate = c

        return [best_candidate] if best_candidate is not None else []

    def draw_pocket_outline(self, image, contours):
        """
        Draws the detected pocket contour as a thin, subtle
        stitch-line.
        """
        for c in contours:
            cv2.drawContours(
                image, [c], -1, (55, 55, 55), 1, lineType=cv2.LINE_AA
            )
        return image

    ###########################################################################
    def draw_shoulder_seam(self, image, shirt_mask, shoulder_frac=0.10):
        """
        Estimates and draws the shoulder seam line from the mask's
        geometry (shoulder_frac percent below the collar, with a
        slight V-shaped dip).
        """
        mask = (shirt_mask > 0).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return image

        y_top, y_bottom = ys.min(), ys.max()
        shirt_h = y_bottom - y_top

        y_shoulder = y_top + int(shirt_h * shoulder_frac)
        row_xs = np.where(mask[y_shoulder] > 0)[0]

        if len(row_xs) == 0:
            return image

        x_left, x_right = row_xs.min(), row_xs.max()
        x_mid = (x_left + x_right) // 2
        dip = int(shirt_h * 0.02)

        pts = np.array([
            [x_left, y_shoulder],
            [x_mid, y_shoulder + dip],
            [x_right, y_shoulder]
        ], dtype=np.int32)

        cv2.polylines(
            image, [pts.reshape(-1, 1, 2)], False,
            (60, 60, 60), 1, lineType=cv2.LINE_AA
        )

        return image

    ####################################################################
    # APPLY SHIRT MASK
    ####################################################################
    @log_execution_time
    def apply_shirt_mask(self, person_image, shirt_mask, prepared_fabric):
        """
        Uses a feathered (soft) alpha-blend instead of a binary
        threshold -> removes the broken/dotted line at the shirt's
        edge, giving a smooth, clean edge.
        """
        mask_f = shirt_mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f = mask_f / 255.0

        mask_f = cv2.GaussianBlur(mask_f, (0, 0), 1.5)
        mask_3ch = cv2.merge([mask_f, mask_f, mask_f])

        output = (
                prepared_fabric.astype(np.float32) * mask_3ch +
                person_image.astype(np.float32) * (1 - mask_3ch)
        )

        return np.clip(output, 0, 255).astype(np.uint8)

    ####################################################################
    # COMPLETE FABRIC RENDER
    ####################################################################
    @log_execution_time
    def render(
            self,
            person_image,
            shirt_mask,
            fabric_info,
            garment_type,
            box,
    ):
        """
        Complete fabric rendering pipeline.

        Steps
        -----
        1. Prepare fabric (period-aligned seamless tiling).
        2. RTV-based structure/fold extraction.
        3. Separate real folds from residual texture (busyness-aware).
        4. Enhance fold contrast (edge-aware).
        5. Apply shading multiplicatively on Lab L-channel only
           (highlight-safe).
        6. Replace shirt region.
        7. Detect + draw buttons.
        8. Detect + draw pocket outline.
        """

        # ----------------------------------------------------------
        # Debug Output Folder
        # ----------------------------------------------------------

        BASE_DIR = Path(__file__).resolve().parents[2]

        DEBUG_FOLDER = BASE_DIR / "test_images" / "debug"

        DEBUG_FOLDER.mkdir(parents=True, exist_ok=True)

        fabric_image = fabric_info["original_fabric"]
        cv2.imwrite(
            str(DEBUG_FOLDER / "debug_0_before_prepared_fabric2.png"),
            fabric_image
        )

        pattern_repeat = fabric_info.get("pattern_repeat")
        pattern_repeat_y = fabric_info.get("pattern_repeat_y")

        logger.info(
            "Before prepare | fabric shape: %s | fabric dtype: %s | "
            "pattern repeat (x): %s | pattern repeat (y): %s",
            fabric_image.shape, fabric_image.dtype, pattern_repeat, pattern_repeat_y
        )

        # ----------------------------------------------------------
        # Step 1: Period-aligned seamless tiling (VirtualFabric)
        # ----------------------------------------------------------

        # prepared_fabric = self.prepare_fabric(
        #     fabric_image=fabric_image,
        #     target_width=person_image.shape[1],
        #     target_height=person_image.shape[0],
        #     repeat_size=pattern_repeat,
        #     repeat_size_y=pattern_repeat_y
        # )
        logger.info("Fabric preparation started")
        prepared_fabric = fit_fabric_to_bbox(
            person_image=person_image,
            fabric_image=fabric_image,
            groundingdino_bbox_xyxy=box
        )
        logger.info("Fabric preparation completed")

        logger.info(
            "After prepare | prepared fabric shape: %s | prepared dtype: %s",
            prepared_fabric.shape, prepared_fabric.dtype
        )

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug_1_after_prepared_fabric2.png"),
            prepared_fabric
        )

        # ----------------------------------------------------------
        # Step 2: preserve_lighting()'s separate multiply is
        # intentionally SKIPPED here -> the RTV shading pipeline
        # below already handles illumination + folds. Applying both
        # was compounding brightness and clipping light fabrics to
        # white. (preserve_lighting() itself has since been removed
        # as dead code.)
        # ----------------------------------------------------------

        realistic_fabric = prepared_fabric

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug_2_after_lighting2.png"),
            realistic_fabric
        )

        # ----------------------------------------------------------
        # Step 3: Estimate how "busy" the original shirt print is.
        # ----------------------------------------------------------

        if garment_type == 'shirt':
            # Step 3: compute a local busyness map for each pixel
            busyness_map = self.estimate_shirt_busyness_map(person_image, shirt_mask)

            # scalar busyness = just the average over the shirt-region
            # pixels (needed for the RTV sigma boost --
            # extract_structure_map_rtv needs a scalar)
            mask_bool = shirt_mask > 0
            busyness_scalar = float(np.mean(busyness_map[mask_bool])) if mask_bool.sum() > 0 else 0.0

            logger.info("Shirt busyness (scalar): %.4f", busyness_scalar)

            # Step 4: get the repeat of the ORIGINAL shirt's checks
            # (not the new fabric's)
            orig_repeat_x, orig_repeat_y = self.estimate_original_shirt_pattern_repeat(
                person_image, shirt_mask
            )
            logger.info(
                "Original shirt pattern repeat (x, y): %s, %s",
                orig_repeat_x, orig_repeat_y
            )

            # NOTE: this recomputes the same value as busyness_scalar
            # above -- flagged as a redundant/duplicate computation in
            # the original code, left untouched since this pass is
            # comments/logging only, not logic.
            busyness_scal = float(np.mean(busyness_map[mask_bool]))
            logger.info("Busyness scalar (duplicate computation): %.4f", busyness_scal)

            # Step 5: RTV extraction -- uses the scalar busyness for
            # the sigma boost
            shading_map = self.extract_structure_map_rtv(
                person_image, shirt_mask,
                pattern_repeat=orig_repeat_x,
                busyness=busyness_scalar
            )

            cv2.imwrite(
                str(DEBUG_FOLDER / "debug_2.1_after_RVT.png"),
                self._debug_visualize_map(shading_map)
            )
            # Step 6: uses the per-pixel busyness_map while suppressing
            # fine texture
            shading_map = self.separate_real_folds_from_texture(
                shading_map,
                busyness_map=busyness_map,
                large_fold_radius=25,
                pattern_pitch=orig_repeat_x
            )

            cv2.imwrite(
                str(DEBUG_FOLDER / "debug_2.2_after_seperate_fold_from_structure.png"),
                self._debug_visualize_map(shading_map)
            )

            # ----------------------------------------------------------
            # Step 6.5 (NEW): Classical intrinsic-decomposition
            # material-edge suppression. Wherever there is a real
            # color/material change (zipper, seam, a differently-colored
            # panel), the shading there is forcibly neutralized (set to
            # 1.0) -- because that isn't real shading, it's a color
            # change belonging to the garment itself, and it will
            # automatically go away once the new fabric replaces it.
            # ----------------------------------------------------------

            chroma_weight = self.estimate_material_edge_mask(
                person_image, shirt_mask
            )
            flat_patch_weight = self.estimate_flat_patch_mask(
                person_image, shirt_mask
            )

            # uses whichever weight is higher between the two signals
            # -- chroma catches colored material changes, flat_patch
            # catches achromatic logo/trim
            material_edge_weight = np.maximum(chroma_weight, flat_patch_weight)

            # debug: saves chroma and flat_patch separately, so in the
            # future it's easy to see at a glance which signal is
            # firing more aggressively.
            cv2.imwrite(
                str(DEBUG_FOLDER / "debug_2.3a_chroma_weight.png"),
                self._debug_visualize_map(chroma_weight)
            )
            cv2.imwrite(
                str(DEBUG_FOLDER / "debug_2.3b_flat_patch_weight.png"),
                self._debug_visualize_map(flat_patch_weight)
            )

            # ----------------------------------------------------------
            # (Issue #14, Round 3 fix): Round 1's coverage-based blanket
            # scale-down turned out to be a "blunt instrument" -- it
            # multiplied the entire weight map by a single flat factor,
            # so suppression on real localized motifs got reduced by
            # the same amount as suppression on a false widespread
            # glossy-artifact (debug_2.1 itself proved that motifs are
            # baked into the shading right from the RTV stage -- they
            # needed maximum suppression strength there, not less).
            #
            # Now it filters by connected-component size instead:
            # - A real localized defect (motif/zipper/seam/trim) always
            #   shows up as a small, scattered blob.
            # - A false positive like glossy-sheen/moiré shows up as
            #   one or a few large blobs covering almost the entire
            #   garment.
            #
            # Small blobs (real motifs) keep getting suppressed at full
            # strength; large blobs (false artifacts) get toned down.
            # ----------------------------------------------------------
            binary_high = (material_edge_weight > 0.5).astype(np.uint8)

            # ----------------------------------------------------------
            # (Issue #14, Round 4 fix): Found the exact root cause of
            # the "wavy pattern" -- in Round 3's component-filtering,
            # the size threshold (3%) could not recognize a dense
            # checkerboard/moiré grid as a single large component,
            # because every small square/diamond in the grid was its
            # own separate connected component -- each individually
            # smaller than 3%. So the whole grid slipped through, and
            # the same regular pattern of suppressed-vs-unsuppressed
            # squares got baked back into the final shading (via the
            # suppression itself) -- this is exactly the "wavy
            # pattern".
            #
            # Now, purely for classification (just to make the
            # decision, not for the actual suppression), an adaptive
            # morphological CLOSING is applied first -- small blobs
            # that are close together (the grid's squares) get merged
            # into one big component, which then correctly crosses the
            # 3% threshold and gets suppressed.
            #
            # Real, sparse motifs (far apart from each other) are not
            # merged by this closing -- they stay as separate, small
            # components, and their full-strength suppression continues
            # to work as before.
            # ----------------------------------------------------------

            ys_shirt, xs_shirt = np.where(mask_bool)
            crop_min_dim = max(
                1, min(ys_shirt.max() - ys_shirt.min(), xs_shirt.max() - xs_shirt.min())
            ) if ys_shirt.size > 0 else 1

            closing_size = max(15, int(crop_min_dim * 0.05))
            if closing_size % 2 == 0:
                closing_size += 1
            closing_kernel = np.ones((closing_size, closing_size), np.uint8)

            merged_binary = cv2.morphologyEx(
                binary_high, cv2.MORPH_CLOSE, closing_kernel
            )

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                merged_binary, connectivity=8
            )

            shirt_area = float(mask_bool.sum())
            MAX_COMPONENT_FRACTION = 0.03  # a single real defect never
                                            # shows up as a blob bigger
                                            # than about 3% of the garment
            LARGE_COMPONENT_SUPPRESS = 0.15  # how much to tone down
                                              # large (false) blobs

            large_components_found = 0

            for label_id in range(1, num_labels):  # 0 = background
                component_area = stats[label_id, cv2.CC_STAT_AREA]
                fraction = component_area / max(shirt_area, 1.0)

                if fraction > MAX_COMPONENT_FRACTION:
                    # suppression is applied only to pixels that were
                    # genuinely high-weight originally (before
                    # closing) -- not to the "gaps" that closing
                    # filled in. This keeps the suppression edges from
                    # becoming artificially wide/blocky.
                    component_region = (labels == label_id)
                    target_pixels = component_region & (binary_high.astype(bool))
                    material_edge_weight[target_pixels] *= LARGE_COMPONENT_SUPPRESS
                    large_components_found += 1

            logger.info(
                "material_edge_weight component-filtered (closing_size=%s) -> "
                "%s merged components found, %s large (>%.0f%% of shirt) "
                "suppressed to %.0f%% strength",
                closing_size, num_labels - 1, large_components_found,
                MAX_COMPONENT_FRACTION * 100, LARGE_COMPONENT_SUPPRESS * 100
            )

            logger.info(
                "material_edge_weight (final) -> mean over shirt: %.4f",
                float(np.mean(material_edge_weight[mask_bool]))
            )
            shading_map = 1.0 + (shading_map - 1.0) * (1.0 - material_edge_weight)

            cv2.imwrite(
                str(DEBUG_FOLDER / "debug_2.3_after_material_edge_suppression.png"),
                self._debug_visualize_map(shading_map)
            )

            # ----------------------------------------------------------
            # Step 6.6 (NEW): Remove the remaining fine grain
            # ----------------------------------------------------------
            shading_map = self.remove_residual_fine_grain(shading_map, shirt_mask)

            cv2.imwrite(
                str(DEBUG_FOLDER / "debug_2.4_after_grain_cleanup.png"),
                self._debug_visualize_map(shading_map)
            )
            # ----------------------------------------------------------
            # Step 6: Edge-aware fold contrast enhancement.
            # ----------------------------------------------------------

            shading_map = self.enhance_fold_contrast(
                shading_map,
                shirt_mask=shirt_mask,
                edge_gain=2.3,
                smooth_sigma=5
            )

            logger.info(
                "Shading map range after enhancement | min: %s | max: %s | std: %s",
                shading_map.min(), shading_map.max(), shading_map.std()
            )

            cv2.imwrite(
                str(DEBUG_FOLDER / "debug_3_rtv_shading_map6.png"),
                np.clip(shading_map * 127, 0, 255).astype(np.uint8)
            )

            # ----------------------------------------------------------
            # Step 7: Apply shading on the Lab L-channel only
            # (highlight-safe).
            # ----------------------------------------------------------

            realistic_fabric = self.apply_structure_map_lab(
                realistic_fabric,
                shading_map
            )

            cv2.imwrite(
                str(DEBUG_FOLDER / "debug_4_after_realistic_fabric2.png"),
                realistic_fabric
            )

        # ----------------------------------------------------------
        # Step 8: Replace only the shirt region.
        # ----------------------------------------------------------

        output = self.apply_shirt_mask(
            person_image,
            shirt_mask,
            realistic_fabric
        )

        # ----------------------------------------------------------
        # Step 9: Buttons.
        # ----------------------------------------------------------

        # buttons = self.detect_buttons(person_image, shirt_mask)
        # logger.info("Buttons found: %s", len(buttons))
        #
        # for (bx, by, br) in buttons:
        #     output = self.draw_synthetic_button(output, bx, by, br)

        # ----------------------------------------------------------
        # Step 10: Pocket outline.
        # ----------------------------------------------------------

        # pocket_contours = self.detect_pocket_outline(person_image, shirt_mask)
        # logger.info("Pocket candidates found: %s", len(pocket_contours))

        # output = self.draw_pocket_outline(output, pocket_contours)

        # ----------------------------------------------------------
        # NEW: Add the placket line and shoulder seam
        # ----------------------------------------------------------
        # output = self.draw_placket_line(output, shirt_mask)
        # output = self.draw_shoulder_seam(output, shirt_mask, shoulder_frac=0.06)

        return output