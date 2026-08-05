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
        Extract a multiplicative shading map using Relative Total
        Variation (RTV) smoothing - separates fold/lighting structure
        (aperiodic) from weave/print texture (periodic), unlike a
        plain frequency-only high-pass filter.

        sigma is tied to the detected weave/print pitch when known;
        for busy (print-heavy) shirts, smoothing is made more
        aggressive to suppress texture leak.
        """

        sigma = 3.0
        lam = 0.015

        if pattern_repeat:
            try:
                pitch = min(pattern_repeat) if isinstance(pattern_repeat, (tuple, list)) else pattern_repeat
                if pitch and pitch > 0:
                    sigma = max(1.5, min(8.0, pitch * 0.5))
            except Exception:
                pass

        # busy प्रिंटसाठी जास्त aggressive smoothing
        if busyness > 0.5:
            sigma = max(sigma, 4.5)
            lam = 0.025

        shading_map = extract_rtv_structure(
            person_image, shirt_mask, sigma=sigma, lam=lam, iterations=5
        )
        return shading_map

    ####################################################################
    # SEPARATE REAL FOLDS FROM RESIDUAL TEXTURE
    ####################################################################

    def separate_real_folds_from_texture(
            self,
            shading_map,
            busyness=0.0,
            large_fold_radius=25
    ):
        """
        Splits the RTV shading map into two bands:

        1. large_scale   -> real folds/wrinkles/drape (extracted with
                             an edge-preserving bilateral filter, NOT
                             a plain Gaussian blur, so true fold
                             ridges stay sharp instead of turning
                             into soft blobs).
        2. fine_residual -> whatever is left (mostly leftover print
                             texture on busy shirts). Only this band
                             is suppressed, based on busyness.

        This keeps genuine fold detail intact while still removing
        residual print leak on busy shirts.
        """

        map_min, map_max = shading_map.min(), shading_map.max()
        norm = (shading_map - map_min) / (map_max - map_min + 1e-6)
        norm_u8 = (norm * 255).astype(np.uint8)

        large_scale_u8 = cv2.bilateralFilter(
            norm_u8,
            d=0,
            sigmaColor=30,
            sigmaSpace=large_fold_radius
        )

        large_scale = (large_scale_u8.astype(np.float32) / 255.0) * \
                      (map_max - map_min) + map_min

        fine_residual = shading_map - large_scale

        fine_blend = float(np.interp(busyness, [0.0, 1.0], [0.9, 0.10]))
        fine_residual = fine_residual * fine_blend

        result = large_scale + fine_residual

        return result

    ####################################################################
    # ENHANCE FOLD CONTRAST (edge-aware unsharp masking)
    ####################################################################

    def enhance_fold_contrast(
            self,
            shading_map,
            edge_gain=1.8,
            smooth_sigma=8,
            clip_range=(0.75, 1.30)
    ):
        """
        Unsharp-masking based contrast enhancement:
        only amplifies LOCAL EDGES (real fold transitions), leaves
        flat regions untouched -> looks like a crease, not a blob.

        Uses a soft (tanh) clip instead of a hard clip, and a light
        final blur, to avoid harsh/plastic-looking edges.
        """

        very_smooth = cv2.GaussianBlur(shading_map, (0, 0), smooth_sigma)

        edge_component = shading_map - very_smooth

        enhanced = very_smooth + edge_component * edge_gain

        lo, hi = clip_range
        center = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0
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

        # गडद करताना (shading < 1) -> साधा multiply पुरेसा आहे
        L_out[~brighten_mask] = L[~brighten_mask] * shading_map[~brighten_mask]

        # उजळ करताना (shading >= 1) -> screen-style soft blend
        excess = shading_map[brighten_mask] - 1.0
        L_out[brighten_mask] = 255 - (255 - L[brighten_mask]) * (1.0 - excess * 0.2)

        L_out = np.clip(L_out, 0, 255)

        lab[:, :, 0] = L_out

        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        return result

    ####################################################################
    # SHIRT BUSYNESS ESTIMATION
    ####################################################################

    def estimate_shirt_busyness(self, person_image, shirt_mask):
        """
        मूळ शर्टाची प्रिंट किती गुंतागुंतीची (busy) आहे ते मोजतं.
        जास्त busyness -> जास्त शक्यता की RTV मध्ये प्रिंट लीक होईल.
        Returns: 0.0 (साधा/plain शर्ट) ते 1.0 (खूप busy प्रिंट)
        """
        gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)
        mask = (shirt_mask > 0).astype(np.uint8)

        if mask.sum() == 0:
            return 0.0

        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        lap_masked = lap[mask > 0]

        busyness = np.std(lap_masked)

        # हे threshold तुमच्या test images वर calibrate करा
        busyness_norm = float(np.clip(busyness / 40.0, 0.0, 1.0))

        return busyness_norm

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
            param2=25,
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

    ####################################################################
    # APPLY SHIRT MASK
    ####################################################################

    def apply_shirt_mask(
            self,
            person_image,
            shirt_mask,
            prepared_fabric
    ):
        """
        Apply the shirt mask to the prepared fabric.

        Purpose
        -------
        This function copies the fabric only inside the detected
        shirt region.

        Everything outside the shirt remains exactly the same.
        """

        binary_mask = (shirt_mask > 0).astype("uint8") * 255

        binary_mask = cv2.merge([
            binary_mask,
            binary_mask,
            binary_mask
        ])

        output = person_image.copy()

        output[binary_mask == 255] = prepared_fabric[binary_mask == 255]

        return output

    ####################################################################
    # COMPLETE FABRIC RENDER
    ####################################################################

    def render(
            self,
            person_image,
            shirt_mask,
            fabric_info,
            fabric_mode="tile"
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
            str(DEBUG_FOLDER / "debug_0_prepared_fabric2.png"),
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

        prepared_fabric = self.prepare_fabric(
            fabric_image=fabric_image,
            target_width=person_image.shape[1],
            target_height=person_image.shape[0],
            repeat_size=pattern_repeat,
            repeat_size_y=pattern_repeat_y
        )

        print("After prepare:", prepared_fabric.shape)
        print("prepared dtype:", prepared_fabric.dtype)

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug_1_prepared_fabric2.png"),
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

        busyness = self.estimate_shirt_busyness(
            person_image,
            shirt_mask
        )

        print("Shirt busyness score:", busyness)

        # ----------------------------------------------------------
        # Step 4: RTV structure extraction.
        # ----------------------------------------------------------

        shading_map = self.extract_structure_map_rtv(
            person_image,
            shirt_mask,
            pattern_repeat=pattern_repeat,
            busyness=busyness
        )

        # ----------------------------------------------------------
        # Step 5: Separate real folds (large_scale) from residual
        # print-leak texture (fine_residual); only suppress the
        # latter, based on busyness.
        # ----------------------------------------------------------

        shading_map = self.separate_real_folds_from_texture(
            shading_map,
            busyness=busyness,
            large_fold_radius=25
        )

        # ----------------------------------------------------------
        # Step 6: Edge-aware fold contrast enhancement.
        # ----------------------------------------------------------

        shading_map = self.enhance_fold_contrast(
            shading_map,
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

        buttons = self.detect_buttons(person_image, shirt_mask)
        print("Buttons found:", len(buttons))

        for (bx, by, br) in buttons:
            output = self.draw_synthetic_button(output, bx, by, br)

        # ----------------------------------------------------------
        # Step 10: Pocket outline.
        # ----------------------------------------------------------

        pocket_contours = self.detect_pocket_outline(person_image, shirt_mask)
        print("Pocket candidates found:", len(pocket_contours))

        output = self.draw_pocket_outline(output, pocket_contours)

        return output