"""
File:
    api/ai/fabric.py

Purpose:
    This module prepares a fabric image for shirt replacement and
    renders it realistically onto the detected shirt region.

Current Milestone:
    ----------------
    ✔ Validate all inputs
    ✔ Period-aligned seamless tiling of the uploaded fabric
    ✔ RTV-based structure/fold extraction (texture-leak free)
    ✔ Lab L-channel-only, highlight-safe shading application
    ✔ Button detection + synthetic button rendering
    ✔ Pocket outline detection (strict, single-candidate)

CHANGELOG (most recent first)
------------------------------
NEW: render() now calls self.prepare_fabric() again (period-aligned
     tiling via VirtualFabric), instead of using the raw fabric
     image directly. Also passes pattern_repeat_y.
NEW: preserve_lighting()'s multiply is NO LONGER applied in render()
     -> it was double-brightening the fabric together with the RTV
     shading map (illumination applied twice, compounding and
     clipping to white on light fabrics). The function is kept
     in the class for reference / future use, just not called.
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

import cv2
import numpy as np
from api.ai.virtual_fabric import VirtualFabric
from pathlib import Path
from api.ai.rtv_smoothing import extract_rtv_structure
from api.ai.fabric_downsampling import fit_fabric_to_bbox
from api.ai.fabric_analyzer import FabricAnalyzer


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
    # INPUT VALIDATION
    ####################################################################

    def validate_inputs(self, person_image, shirt_mask, fabric_image):
        """
        Validate every input before processing.

        Parameters
        ----------
        person_image : numpy.ndarray
            Original image uploaded by the user.

        shirt_mask : numpy.ndarray
            Binary mask returned by SAM.

        fabric_image : numpy.ndarray
            Uploaded fabric texture.

        Raises
        ------
        ValueError
            If any input is invalid.
        """

        if person_image is None:
            raise ValueError("Person image is None.")

        if shirt_mask is None:
            raise ValueError("Shirt mask is None.")

        if fabric_image is None:
            raise ValueError("Fabric image is None.")

        if person_image.size == 0:
            raise ValueError("Person image is empty.")

        if shirt_mask.size == 0:
            raise ValueError("Shirt mask is empty.")

        if fabric_image.size == 0:
            raise ValueError("Fabric image is empty.")

        # The SAM mask should have the same height and width
        # as the original image.
        if person_image.shape[:2] != shirt_mask.shape[:2]:
            raise ValueError(
                "Mask size does not match person image size."
            )

    ####################################################################
    # RESIZE FABRIC (legacy helper, kept for compatibility)
    ####################################################################

    def resize_fabric(self, fabric_image, target_height, target_width):
        """
        Simple stretch-resize of the fabric. Not used by render()
        anymore (period-aligned tiling via prepare_fabric() is used
        instead), kept here in case a caller still needs a plain
        resize utility.
        """

        resized = cv2.resize(
            fabric_image,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )

        return resized

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
        """

        return self.virtual_fabric.generate(
            fabric_image=fabric_image,
            target_width=target_width,
            target_height=target_height,
            repeat_size=repeat_size,
            repeat_size_y=repeat_size_y
        )

    ####################################################################
    # PRESERVE LIGHTING USING A NORMALIZED LIGHTING MAP
    #
    # NOTE: This function is intentionally NOT called from render()
    # anymore. The RTV-based shading pipeline (extract_structure_map_rtv
    # -> separate_real_folds_from_texture -> enhance_fold_contrast ->
    # apply_structure_map_lab) already handles illumination + folds.
    # Calling both this AND that pipeline compounds the brightness
    # multiplier and blows out light-coloured fabrics to white.
    # Kept here for reference / possible future use.
    ####################################################################

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
        # NEW: autocorrelation fail (None) झाल्यास FFT fallback वापरा
        # ------------------------------------------------------------
        if repeat_x is None:
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            repeat_x = self.fabric_analyzer.detect_pattern_repeat_fft_fallback(gray_crop)
            print("Autocorrelation failed -> FFT fallback repeat_x:", repeat_x)

        return repeat_x, repeat_y
    def preserve_lighting(
            self,
            person_image,
            prepared_fabric
    ):
        """
        Preserve the original shirt lighting while keeping the uploaded
        fabric colours almost unchanged.

        This version extracts only the illumination component and
        applies it to the uploaded fabric.
        """

        fabric = prepared_fabric.astype(np.float32)

        gray = cv2.cvtColor(
            person_image,
            cv2.COLOR_BGR2GRAY
        ).astype(np.float32)

        illumination = cv2.GaussianBlur(
            gray,
            (81, 81),
            0
        )

        illumination = illumination / (illumination.mean() + 1e-6)

        illumination = np.clip(
            illumination,
            0.60,
            1.40
        )

        illumination = illumination[:, :, np.newaxis]

        result = fabric * illumination

        result = np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

        return result

    ####################################################################
    # LEGACY: Gaussian high-pass fold map (superseded by RTV pipeline)
    #
    # Kept for reference / comparison only. Not called from render().
    ####################################################################

    def extract_fold_map(
            self,
            person_image,
            shirt_mask
    ):
        """
        Extract only folds and wrinkles using a Gaussian high-pass
        filter. Superseded by extract_structure_map_rtv(), which
        properly separates texture (periodic) from fold shading
        (aperiodic) instead of relying on frequency alone.
        """

        gray = cv2.cvtColor(
            person_image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.bitwise_and(
            gray,
            gray,
            mask=shirt_mask.astype(np.uint8)
        )

        low_frequency = cv2.GaussianBlur(
            gray,
            (41, 41),
            0
        )

        high_frequency = cv2.subtract(
            gray,
            low_frequency
        )

        fold_map = cv2.normalize(
            high_frequency,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        return fold_map

    def clean_fold_map(
            self,
            fold_map
    ):
        """
        Remove tiny shirt texture while keeping folds.
        (Legacy helper for extract_fold_map(), not used by render().)
        """

        kernel = np.ones((5, 5), np.uint8)

        cleaned = cv2.morphologyEx(
            fold_map,
            cv2.MORPH_OPEN,
            kernel
        )

        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel
        )

        return cleaned

    def apply_fold_map(
            self,
            fabric_image,
            fold_map,
            strength=0.18
    ):
        """
        Apply only fold information (legacy, additive blend).
        Superseded by apply_structure_map_lab() (multiplicative,
        Lab L-channel only). Not used by render().
        """

        fold = fold_map.astype(np.float32)

        fold = (fold - 128.0) / 255.0

        fold = fold[:, :, np.newaxis]

        fabric = fabric_image.astype(np.float32)

        result = fabric + (fold * 255 * strength)

        result = np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

        return result

    ####################################################################
    # RTV STRUCTURE EXTRACTION
    ####################################################################

    def extract_structure_map_rtv(
            self,
            person_image,
            shirt_mask,
            pattern_repeat=None,
            busyness=0.0
    ):
        """
        NEW: sigma cap आता fixed "8.0" नाही, तर शर्ट crop च्या actual size
        वरून adaptive आहे. जुना fixed cap मोठ्या checks (उदा. pitch=38px)
        साठी खूप लहान पडत होता -- त्यामुळे pitch=38 आणि pitch=420 दोन्ही
        सारख्याच sigma=8.0 वर clip होत होते, आणि Fix #1 चा output वर
        काहीच परिणाम दिसत नव्हता.
        """

        mask_bin = (shirt_mask > 0).astype(np.uint8)
        ys, xs = np.where(mask_bin > 0)

        if len(ys) > 0:
            crop_h = ys.max() - ys.min()
            crop_w = xs.max() - xs.min()
            crop_min_dim = max(1, min(crop_h, crop_w))
        else:
            crop_min_dim = 200

        # शर्ट crop च्या 18% पर्यंत sigma जाऊ शकतो (किमान 8.0 राहील,
        # जेणेकरून बारीक weave texture साठी जुनं behavior तसंच राहील)
        sigma_cap = max(8.0, crop_min_dim * 0.18)

        sigma = 3.0
        lam = 0.015

        if pattern_repeat:
            try:
                pitch = min(pattern_repeat) if isinstance(pattern_repeat, (tuple, list)) else pattern_repeat
                if pitch and pitch > 0:
                    sigma = max(1.5, min(sigma_cap, pitch * 0.55))
            except Exception:
                pass

        if busyness > 0.5:
            sigma = max(sigma, sigma_cap * 0.5)
            lam = 0.03

        # ---- हे print जरूर ठेवा -- पुढच्या run मध्ये actual sigma बघून confirm करा ----
        print(
            f"RTV params -> pitch: {pattern_repeat}, crop_min_dim: {crop_min_dim}, "
            f"sigma_cap: {sigma_cap:.2f}, final sigma: {sigma:.2f}, lam: {lam}"
        )

        shading_map = extract_rtv_structure(
            person_image, shirt_mask, sigma=sigma, lam=lam, iterations=6
        )
        return shading_map

        ####################################################################
        # SHIRT BUSYNESS ESTIMATION
        ####################################################################

    def estimate_shirt_busyness_map(self, person_image, shirt_mask, window=15):
        """
        Fixed constant (/40.0) ऐवजी, प्रत्येक फोटोच्या स्वतःच्या actual
        contrast range वरून adaptive normalization करतो. यामुळे वेगवेगळ्या
        exposure/resolution च्या फोटोंवर सुद्धा busyness_map नेहमी पूर्ण
        [0,1] range मध्ये spread होतो -- "calibrate on test images" ही
        गरजच राहत नाही.
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
        # NEW: fixed /40.0 ऐवजी, या फोटोतल्या शर्ट भागातल्या 95th
        # percentile वरून normalize करा -- adaptive, self-calibrating.
        # ------------------------------------------------------------
        ref_high = np.percentile(local_std[mask], 95)
        ref_high = max(ref_high, 1e-3)  # divide-by-zero टाळण्यासाठी

        busyness_map = np.clip(local_std / ref_high, 0.0, 1.0)

        print(
            "Busyness map stats -> ref_high(95th pct):", ref_high,
            "| mean over shirt:", float(np.mean(busyness_map[mask])),
            "| max:", float(np.max(busyness_map[mask]))
        )

        return busyness_map * mask.astype(np.float32)


    ####################################################################
    # CHROMATICITY-BASED MATERIAL EDGE MASK
    # (Classical Intrinsic Image Decomposition -- Retinex-style
    # colour-ratio test. कुठलाही Neural Network नाही, फक्त gradients.)
    ####################################################################

    def estimate_material_edge_mask(self, person_image, shirt_mask):
        """
        NEW (v2): Region-based classical intrinsic decomposition.

        आधीची version फक्त रंग-बदलाच्या CADGE वर काम करत होती (पातळ
        सीमा), त्यामुळे झिपर/पॅनलचा आतला भाग तसाच (detailed) राहायचा.

        आता: शर्ट भागाचा DOMINANT (सगळ्यात जास्त वापरलेला) रंग काढतो,
        आणि प्रत्येक pixel त्या रंगापासून किती दूर आहे ते मोजतो. जो
        संपूर्ण भाग dominant रंगापेक्षा बराच वेगळा आहे (झिपर, वेगळ्या
        रंगाचा पॅनल, trim) -- तो पूर्ण भाग (नुसती कड नाही,
        morphological closing ने आतला भागही भरून) तटस्थ करतो.

        मूळ फॅब्रिक-रंगाच्याच भागातलं RTV shading (म्हणजे खरे folds)
        जसंच्या तसं ठेवतो.
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
        # Dominant fabric रंग -- शर्ट भागातल्या a/b चा MEDIAN (मोठ्या
        # outlier भागांकडे (झिपर, ट्रिम) दुर्लक्ष करतो, कारण मूळ फॅब्रिक
        # भागच बहुसंख्य असतो)
        # ------------------------------------------------------------
        a_dom = float(np.median(a[mask_bool]))
        b_dom = float(np.median(b[mask_bool]))

        chroma_dist = np.sqrt((a - a_dom) ** 2 + (b - b_dom) ** 2)

        # ------------------------------------------------------------
        # NEW: adaptive threshold -- शर्ट भागातल्याच chroma_dist च्या
        # 70th percentile वरून (fixed नाही)
        # ------------------------------------------------------------
        ref_high = np.percentile(chroma_dist[mask_bool], 70)
        ref_high = max(ref_high, 1e-3)

        region_weight = np.clip(chroma_dist / ref_high, 0.0, 1.0)

        # किरकोळ/सौम्य रंग-drift (जो नुसत्या shading मुळेही होऊ शकतो)
        # वगळण्यासाठी squared करतो -- फक्त खरंच वेगळा रंग असलेला भाग
        # जास्त वजन घेईल
        region_weight = region_weight ** 2

        # ------------------------------------------------------------
        # NEW: Morphological CLOSING -- आधीच्या version मध्ये फक्त
        # सीमा पकडली जायची, आता झिपर/पॅनलचा आतला भागही एकसंधपणे भरला
        # जातो (kernel size crop च्या size वरून adaptive)
        # ------------------------------------------------------------
        kernel_size = max(3, int(crop_min_dim * 0.03))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        region_u8 = (region_weight * 255).astype(np.uint8)
        region_u8 = cv2.morphologyEx(region_u8, cv2.MORPH_CLOSE, kernel)
        region_weight = region_u8.astype(np.float32) / 255.0

        # कडेवर मऊ संक्रमण (hard cutoff टाळण्यासाठी)
        blur_sigma = max(2.0, min(15.0, crop_min_dim * 0.03))
        region_weight = cv2.GaussianBlur(region_weight, (0, 0), blur_sigma)

        region_weight = region_weight * mask_bool.astype(np.float32)

        print(
            f"estimate_material_edge_mask -> crop_min_dim: {crop_min_dim}, "
            f"a_dom: {a_dom:.1f}, b_dom: {b_dom:.1f}, ref_high(70th pct): {ref_high:.2f}, "
            f"kernel_size: {kernel_size}, blur_sigma: {blur_sigma:.2f}, "
            f"mean region_weight over shirt: {float(np.mean(region_weight[mask_bool])):.4f}"
        )

        return region_weight



    ####################################################################
    # FLAT-PATCH DETECTOR (achromatic logos/trims — chroma नाही तरी पकडतो)
    ####################################################################

    def estimate_flat_patch_mask(self, person_image, shirt_mask):
        """
        Chroma-based check (a/b) फक्त रंगीत material बदल पकडतो. पण logo/
        trim सारखे भाग बरेचदा राखाडी-वर-राखाडी (achromatic) असतात --
        तिथे a/b जवळपास सेमच राहतो, त्यामुळे chroma check त्यांना सोडून
        देतो.

        फरक असा: खरं fold नेहमी GRADUAL (सतत बदलणारं) असतं. logo/trim/
        patch मात्र एक सपाट (flat) प्रदेश असतो ज्याची सीमा अचानक
        (sharp) असते. हे function दोन वेगळ्या window sizes वरचा L
        (brightness) std काढून हा फरक शोधतो.
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

        # खरं fold: small_std ≈ medium_std (सतत बदल) -> flatness_ratio ≈ 1
        # flat patch: small_std << medium_std (आत सपाट, सीमेवर उडी) -> ratio ≈ 0
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

        print(
            f"estimate_flat_patch_mask -> crop_min_dim: {crop_min_dim}, "
            f"small_win: {small_win}, medium_win: {medium_win}, "
            f"mean plateau_score over shirt: {float(np.mean(plateau_score[mask_bool])):.4f}"
        )

        return plateau_score

    ####################################################################
    # RESIDUAL FINE-GRAIN CLEANUP (शेवटचा polish pass)
    ####################################################################

    def remove_residual_fine_grain(self, shading_map, shirt_mask):
        """
        वरच्या सगळ्या स्टेप्सनंतरही sleeve सारख्या भागात उरलेला बारीक
        grain/noise texture काढण्यासाठी शेवटचा edge-preserving smoothing
        pass. मोठे fold-आकार (large gradients) टिकवतो, फक्त बारीक दाणेदार
        noise काढतो.
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

        print(
            f"remove_residual_fine_grain -> crop_min_dim: {crop_min_dim}, "
            f"grain_radius: {grain_radius:.2f}, grain_sigma_color: {grain_sigma_color:.2f}"
        )

        return smoothed
    ####################################################################
    # DEBUG VISUALIZATION HELPER (adaptive — कुठलाही fixed multiplier नाही)
    ####################################################################

    def _debug_visualize_map(self, value_map):
        """
        shading_map च्या values साधारण 0.3-1.6 range मध्ये (float)
        असतात. cv2.imwrite ला हा array जसाच्या तसा दिला की तो प्रत्येक
        pixel ला जवळपास 0/1 (uint8) असं वाचतो -> संपूर्ण इमेज काळी
        दिसते.

        हे फंक्शन प्रत्येक इमेजच्या स्वतःच्या ACTUAL min/max वरून
        adaptive पद्धतीने 0-255 range मध्ये स्ट्रेच करतं -- त्यामुळे
        प्रत्येक वेगळ्या फोटोवर (वेगळी shading range असली तरी) नेहमी
        योग्य contrast दिसेल. इथे कुठलाही fixed multiplier (जसं आधीचं
        *127) वापरलेला नाही.
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

    def separate_real_folds_from_texture(
            self, shading_map, busyness_map=None, busyness=0.0,
            large_fold_radius=25, pattern_pitch=None
    ):
        """
        NEW: bilateral radius (sigmaSpace) पूर्वीच check/line pitch
        पेक्षा मोठा केला होता (Issue #6).

        NEW (Issue #8 fix): sigmaColor आता FIXED 45 नाही. bilateral
        filter मध्ये sigmaSpace (radius) आणि sigmaColor (intensity
        threshold) हे दोन SWATANTRA axes आहेत -- radius कितीही
        वाढवला तरी, जर check pattern चा intensity-contrast fixed
        sigmaColor पेक्षा जास्त असेल, तर filter त्या check-कडांना
        "real edge" समजून तशाच ठेवतो (blur करत नाही) -- म्हणजे radius
        फिक्स करूनही checks लीक होतच राहतात.

        आता sigmaColor हा शर्ट भागातल्या ACTUAL intensity spread
        (norm_u8 चा std, busyness_map द्वारे मास्क केलेला) वरून
        adaptive काढला जातो -- जास्त busy/high-contrast प्रिंटसाठी
        जास्त sigmaColor, कमी busy शर्टसाठी कमी.
        """

        if pattern_pitch:
            # radius नेहमी pitch च्या किमान 1.8 पट मोठा ठेवा
            large_fold_radius = max(large_fold_radius, int(pattern_pitch * 1.8))

        map_min, map_max = shading_map.min(), shading_map.max()
        norm = (shading_map - map_min) / (map_max - map_min + 1e-6)
        norm_u8 = (norm * 255).astype(np.uint8)

        # ------------------------------------------------------------
        # NEW: sigmaColor adaptive -- शर्ट भागातल्या actual intensity
        # spread वरून (busyness_map>0 = शर्ट pixels, बाकी बाहेरचे नाही)
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

        print(
            f"separate_real_folds -> intensity_std: {intensity_std:.2f}, "
            f"adaptive sigmaColor: {sigma_color:.2f}"
        )

        large_scale_u8 = cv2.bilateralFilter(
            norm_u8, d=0, sigmaColor=sigma_color, sigmaSpace=large_fold_radius
        )
        large_scale = (large_scale_u8.astype(np.float32) / 255.0) * \
                      (map_max - map_min) + map_min

        fine_residual = shading_map - large_scale

        if busyness_map is not None:
            bm = cv2.resize(busyness_map, (shading_map.shape[1], shading_map.shape[0]))
            # जास्त aggressive suppression: आधी 0.9->0.10 (जास्तीत जास्त फक्त
            # ४५% suppress होत होतं), आता 0.7->0.02 (जास्तीत जास्त ~९८% suppress)
            fine_blend_map = 0.7 - (0.7 - 0.02) * bm
            fine_blend_map = np.clip(fine_blend_map, 0.02, 0.7)
            fine_residual = fine_residual * fine_blend_map
        else:
            fine_blend = float(np.interp(busyness, [0.0, 1.0], [0.7, 0.02]))
            fine_residual = fine_residual * fine_blend

        print(
            f"separate_real_folds -> pattern_pitch: {pattern_pitch}, "
            f"large_fold_radius used: {large_fold_radius}"
        )

        return large_scale + fine_residual
    ####################################################################
    # ENHANCE FOLD CONTRAST (edge-aware unsharp masking)
    ####################################################################

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

        NEW (Issue #7 fix): clip_range आता FIXED (0.75, 1.30) नाही.
        जुना fixed clip_range खूप अरुंद असल्यामुळे tanh() नेहमी त्याच
        [0.75, 1.30] बाउंड्सवर saturate होत होता -- upstream मध्ये
        sigma/large_fold_radius कितीही बदलले तरी final output नेहमी
        जवळपास सेम range/std मध्ये दिसत होता (verified via logs:
        min/max दोन्ही run मध्ये 0.75/1.30 च्या अगदी जवळ अडकलेले होते).

        आता clip_range हा शर्ट भागातल्या ACTUAL enhanced values च्या
        std वरून adaptive काढला जातो -- जेवढा jास्त real fold/texture
        signal, तेवढी jास्त room tanh ला मिळेल, जेणेकरून upstream चे
        फिक्स प्रत्यक्ष output मध्ये दिसतील.

        Uses a soft (tanh) clip instead of a hard clip, and a light
        final blur, to avoid harsh/plastic-looking edges.
        """

        very_smooth = cv2.GaussianBlur(shading_map, (0, 0), smooth_sigma)

        edge_component = shading_map - very_smooth

        enhanced = very_smooth + edge_component * edge_gain

        # ------------------------------------------------------------
        # NEW: clip_range शर्ट भागाच्या actual spread वरून adaptive
        # ------------------------------------------------------------
        if clip_range is None:
            if shirt_mask is not None:
                mask_bool = shirt_mask > 0
                sample = enhanced[mask_bool] if mask_bool.sum() > 0 else enhanced
            else:
                sample = enhanced

            measured_std = float(np.std(sample))

            # किमान 0.15 half_range ठेवा (खूप सपाट/plain शर्टवर range
            # शून्याजवळ जाऊन divide-by-near-zero टाळण्यासाठी)
            half_range = max(0.15, measured_std * 3.0)

            center = 1.0
            clip_range = (center - half_range, center + half_range)

        lo, hi = clip_range
        center = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0

        print(
            f"enhance_fold_contrast -> measured_std: {float(np.std(enhanced)):.5f}, "
            f"clip_range used: ({lo:.3f}, {hi:.3f})"
        )

        enhanced = center + half_range * np.tanh(
            (enhanced - center) / half_range
        )

        enhanced = cv2.GaussianBlur(enhanced, (0, 0), 1.2)

        return enhanced
    ####################################################################
    # APPLY STRUCTURE MAP (Lab L-channel, highlight-safe)
    ####################################################################

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

        #newly added ----------------
        shadow_strength = 2.9

        # गडद करताना (shading < 1) -> साधा multiply पुरेसा आहे
        shadow_map = 1.0 - (1.0 - shading_map[~brighten_mask]) * shadow_strength
        shadow_map = np.clip(shadow_map, 0.0, 1.0)

        #---------------------------------------

        L_out[~brighten_mask] = L[~brighten_mask] * shadow_map

        # उजळ करताना (shading >= 1) -> screen-style soft blend
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
        # Non-max suppression -> एकाच बटणासाठी अनेक circles टाळा
        # --------------------------------------------------------------
        deduped = []
        for (x, y, r) in raw:
            too_close = any(
                np.hypot(x - kx, y - ky) < 25 for (kx, ky, kr) in deduped
            )
            if not too_close:
                deduped.append((x, y, r))

        # --------------------------------------------------------------
        # जास्तीत जास्त बटणं मर्यादित करा (Hough आधीच confidence नुसार
        # sorted देतो, त्यामुळे top-N घेणं सुरक्षित आहे)
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

        # फक्त एक हलकी highlight
        cv2.circle(
            overlay, (x - r // 3, y - r // 3), max(1, r // 4),
            (255, 255, 255), -1, lineType=cv2.LINE_AA
        )

        # फक्त 2 सूक्ष्म, low-contrast थ्रेड-होल्स
        hole_r = max(1, r // 6)
        cv2.circle(overlay, (x - r // 4, y), hole_r, (120, 120, 120), -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x + r // 4, y), hole_r, (120, 120, 120), -1, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.82, image, 0.18, 0, dst=image)
        return image
    ############################################################
    def draw_placket_line(self, image, shirt_mask):
        """
        शर्टाच्या मधोमध, बटणांच्या रांगेजवळ एक हलकी dashed शिवण-रेषा काढतो
        -> शर्ट "शिवलेला" वाटतो. पूर्णपणे भौमितिक (mask-आधारित), AI नाही.
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

            # खूप मोठा minimum-area threshold -> fabric texture noise गळून पडतो
            if area < 0.015 * shirt_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h + 1e-6)
            if not (0.7 < aspect < 2.2):
                continue

            rel_y = (y - ys.min()) / float(shirt_h)
            if not (0.18 < rel_y < 0.45):
                continue

            # आकार साधारण आयताकृती आहे का ते तपासा
            approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
            if not (4 <= len(approx) <= 8):
                continue

            # भरीव, घन आकार हवा -- विस्कळीत, तुटक रेषा नको
            fill_ratio = area / float(w * h + 1e-6)
            if fill_ratio < 0.55:
                continue

            # फक्त सर्वात मोठा, सर्वात विश्वासार्ह उमेदवार ठेवा
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
        Mask च्या भूमितीवरून खांद्याची शिवण-रेषा अंदाजे काढतो
        (collar पासून shoulder_frac टक्के खाली, हलकी V-आकाराची झुक).
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

    def apply_shirt_mask(self, person_image, shirt_mask, prepared_fabric):
        """
        NEW: Binary threshold ऐवजी feathered (soft) alpha-blend वापरतो
        -> शर्टाच्या कडेवरची तुटक/ठिपक्यासारखी रेषा जाते, मऊ स्वच्छ कड मिळते.
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

        print("Before prepare:", fabric_image.shape)
        print("Fabric dtype:", fabric_image.dtype)
        print("Pattern repeat (x):", pattern_repeat)
        print("Pattern repeat (y):", pattern_repeat_y)

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
        print("Fabric is preparing ...........")
        prepared_fabric=fit_fabric_to_bbox(
            person_image= person_image,
            fabric_image=fabric_image,
            groundingdino_bbox_xyxy=box
        )
        print("Fabric is prepared 🧣👍")

        print("After prepare:", prepared_fabric.shape)
        print("prepared dtype:", prepared_fabric.dtype)

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug_1_after_prepared_fabric2.png"),
            prepared_fabric
        )

        # ----------------------------------------------------------
        # Step 2: preserve_lighting()'s separate multiply is
        # intentionally SKIPPED here -> the RTV shading pipeline
        # below already handles illumination + folds. Applying both
        # was compounding brightness and clipping light fabrics to
        # white.
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
            # Step 3: प्रत्येक pixel साठी local busyness map काढा
            busyness_map = self.estimate_shirt_busyness_map(person_image, shirt_mask)

            # scalar busyness = फक्त शर्ट भागातल्या pixels ची सरासरी
            # (RTV sigma boost साठी लागतो -- extract_structure_map_rtv ला scalar हवा)
            mask_bool = shirt_mask > 0
            busyness_scalar = float(np.mean(busyness_map[mask_bool])) if mask_bool.sum() > 0 else 0.0

            print("Shirt busyness (scalar):", busyness_scalar)

            # Step 4: मूळ शर्टाच्या checks चा repeat काढा (नवीन fabric चा नाही)
            orig_repeat_x, orig_repeat_y = self.estimate_original_shirt_pattern_repeat(
                person_image, shirt_mask
            )
            print("Original shirt pattern repeat (x, y):", orig_repeat_x, orig_repeat_y)

            busyness_scal = float(np.mean(busyness_map[mask_bool]))
            print(">>> busyness_scalar:", busyness_scal)

            # Step 5: RTV extraction -- scalar busyness sigma boost साठी वापरतो
            shading_map = self.extract_structure_map_rtv(
                person_image, shirt_mask,
                pattern_repeat=orig_repeat_x,
                busyness=busyness_scalar  # <-- आता scalar व्यवस्थित मिळतो
            )

            cv2.imwrite(
                str(DEBUG_FOLDER / "debug_2.1_after_RVT.png"),
                self._debug_visualize_map(shading_map)
            )
            # Step 6: fine-texture suppress करताना per-pixel busyness_map वापरतो
            shading_map = self.separate_real_folds_from_texture(
                shading_map,
                busyness_map=busyness_map,
                large_fold_radius=25,
                pattern_pitch=orig_repeat_x  # <-- NEW: 38 इथे जाईल
            )

            cv2.imwrite(
                str(DEBUG_FOLDER / "debug_2.2_after_seperate_fold_from_structure.png"),
                self._debug_visualize_map(shading_map)
            )

            # ----------------------------------------------------------
            # Step 6.5 (NEW): Classical intrinsic-decomposition
            # material-edge suppression. जिथे खरा रंग/material बदल आहे
            # (zipper, seam, वेगळा रंग-पॅनल) तिथली shading जबरदस्तीने
            # तटस्थ (1.0) करतो -- कारण ते खरं शेडिंग नसून गारमेंटचाच
            # रंग-बदल आहे, आणि तो नवीन fabric replace केल्यावर आपोआप
            # निघून जाणार आहे.
            # ----------------------------------------------------------

            chroma_weight = self.estimate_material_edge_mask(
                person_image, shirt_mask
            )
            flat_patch_weight = self.estimate_flat_patch_mask(
                person_image, shirt_mask
            )

            # दोन्ही signal मधलं जास्त असलेलं वजन वापरतो -- रंगीत material
            # बदल chroma पकडतो, achromatic logo/trim flat_patch पकडतो
            material_edge_weight = np.maximum(chroma_weight, flat_patch_weight)

            shading_map = 1.0 + (shading_map - 1.0) * (1.0 - material_edge_weight)

            cv2.imwrite(
                str(DEBUG_FOLDER / "debug_2.3_after_material_edge_suppression.png"),
                self._debug_visualize_map(shading_map)
            )

            # ----------------------------------------------------------
            # Step 6.6 (NEW): उरलेला fine grain काढा
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

            print(
                "Shading map range after enhancement:",
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
        # print("Buttons found:", len(buttons))
        #
        # for (bx, by, br) in buttons:
        #     output = self.draw_synthetic_button(output, bx, by, br)

        # ----------------------------------------------------------
        # Step 10: Pocket outline.
        # ----------------------------------------------------------

        # pocket_contours = self.detect_pocket_outline(person_image, shirt_mask)
        # print("Pocket candidates found:", len(pocket_contours))

        # output = self.draw_pocket_outline(output, pocket_contours)

        # ----------------------------------------------------------
        # NEW: Placket line आणि shoulder seam जोडा
        # ----------------------------------------------------------
        # output = self.draw_placket_line(output, shirt_mask)
        # output = self.draw_shoulder_seam(output, shirt_mask, shoulder_frac=0.06)

        return output