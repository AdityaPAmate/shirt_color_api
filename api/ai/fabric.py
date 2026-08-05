"""
File:
    api/ai/fabric.py

Purpose:
    This module prepares a fabric image for shirt replacement.

Current Milestone:
    ----------------
    ✔ Validate all inputs
    ✔ Resize or tile the uploaded fabric
    ✔ Return a prepared fabric image

Next Milestone:
    ----------------
    Apply the shirt mask and preserve lighting.
"""

import cv2
import numpy as np
from api.ai.virtual_fabric import VirtualFabric
from pathlib import Path
from api.ai.rtv_smoothing import extract_rtv_structure

class FabricRenderer:
    """
    FabricRenderer is responsible for preparing the uploaded fabric
    before it is applied to the detected shirt.

    This class DOES NOT:
        - detect the shirt
        - generate the SAM mask
        - save images

    It only prepares the fabric image.

    Future milestones will extend this class to perform
    realistic shirt rendering.
    """

    def __init__(self):
        """
        Initialize helper classes.

        Currently we only initialize the
        Virtual Fabric generator.

        In future more helper classes
        will be initialized here.
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
    # RESIZE FABRIC
    ####################################################################

    def resize_fabric(self, fabric_image, target_height, target_width):
        """
        Resize the uploaded fabric to match the person's image size.

        Parameters
        ----------
        fabric_image : numpy.ndarray

        target_height : int

        target_width : int

        Returns
        -------
        numpy.ndarray
            Resized fabric image.
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
            repeat_size_y=None          # NEW
    ):
        return self.virtual_fabric.generate(
            fabric_image=fabric_image,
            target_width=target_width,
            target_height=target_height,
            repeat_size=repeat_size,
            repeat_size_y=repeat_size_y  # NEW
        )
    ####################################################################
    # PRESERVE ORIGINAL SHIRT LIGHTING
    ####################################################################

    ####################################################################
    # PRESERVE LIGHTING USING A NORMALIZED LIGHTING MAP
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

        # ---------------------------------------------
        # Convert images to float
        # ---------------------------------------------
        fabric = prepared_fabric.astype(np.float32)

        gray = cv2.cvtColor(
            person_image,
            cv2.COLOR_BGR2GRAY
        ).astype(np.float32)

        # ---------------------------------------------
        # Estimate illumination
        # ---------------------------------------------
        illumination = cv2.GaussianBlur(
            gray,
            (81, 81),
            0
        )

        # ---------------------------------------------
        # Normalize illumination
        # ---------------------------------------------
        illumination = illumination / (illumination.mean() + 1e-6)

        # ---------------------------------------------
        # Keep stronger lighting variation
        # ---------------------------------------------
        illumination = np.clip(
            illumination,
            0.60,
            1.40
        )

        illumination = illumination[:, :, np.newaxis]

        # ---------------------------------------------
        # Apply lighting
        # ---------------------------------------------
        result = fabric * illumination

        result = np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

        return result

    def extract_fold_map(
            self,
            person_image,
            shirt_mask
    ):
        """
        Extract only folds and wrinkles.

        This removes most of the original shirt texture.
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

        # Large illumination
        low_frequency = cv2.GaussianBlur(
            gray,
            (41, 41),
            0
        )

        # High-pass image
        high_frequency = cv2.subtract(
            gray,
            low_frequency
        )

        # Normalize
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
        Apply only fold information.

        Original shirt colour and pattern are not copied.
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

    def extract_structure_map_rtv(self, person_image, shirt_mask, pattern_repeat=None, busyness=0.0):
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
            sigma = max(sigma, 4.5)  # आधी 6.0 होता, आता कमी केला
            lam = 0.025  # आधी 0.035 होता, आता कमी केला

        shading_map = extract_rtv_structure(
            person_image, shirt_mask, sigma=sigma, lam=lam, iterations=5
        )
        return shading_map

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

    def separate_real_folds_from_texture(
            self,
            shading_map,
            busyness=0.0,
            large_fold_radius=25  # आधी 45 होता -> कमी केला, edges जास्त शाबूत राहतील
    ):
        # ------------------------------------------------------------
        # Gaussian ऐवजी bilateral -> edges (खरे fold ridges) जपले जातात
        # ------------------------------------------------------------
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

    def enhance_fold_contrast(
            self,
            shading_map,
            edge_gain=1.8,
            smooth_sigma=8,
            clip_range=(0.75, 1.30)
    ):
        """
        Global gain ऐवजी unsharp masking वापरतो:
        फक्त local edges (खरे fold transitions) तीक्ष्ण करतो,
        सपाट भागांना touch करत नाही -> blob ऐवजी crease सारखं दिसतं.
        """
        # खूप coarse (मोठ्या स्केलचं) आवृत्ती काढा
        very_smooth = cv2.GaussianBlur(shading_map, (0, 0), smooth_sigma)

        # local edge component = जिथे अचानक बदल आहे तिथेच value मिळेल
        edge_component = shading_map - very_smooth

        # फक्त edge component amplify करा, बाकी coarse base तसाच ठेवा
        enhanced = very_smooth + edge_component * edge_gain

        # Soft clip -> harsh cutoff टाळण्यासाठी
        lo, hi = clip_range
        center = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0
        enhanced = center + half_range * np.tanh(
            (enhanced - center) / half_range
        )

        enhanced = cv2.GaussianBlur(enhanced, (0, 0), 1.2)

        return enhanced

    ################################################################
    #Buttons
    ##################################################################

    def detect_buttons(self, person_image, shirt_mask, max_buttons=8):
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
            minDist=30,  # आधी 22 होता -> वाढवला, जवळपासची duplicates टाळण्यासाठी
            param1=60,
            param2=25,  # आधी 18 होता -> वाढवला, weak/false detections कमी होतील
            minRadius=5,
            maxRadius=11
        )

        if circles is None:
            return []

        raw = [(int(c[0]), int(c[1]), int(c[2])) for c in circles[0]]
        raw = [c for c in raw if strip_mask[c[1], c[0]] > 0]

        # --------------------------------------------------------------
        # NEW: Non-max suppression -> एकाच बटणासाठी अनेक circles टाळा
        # --------------------------------------------------------------
        deduped = []
        for (x, y, r) in raw:
            too_close = any(
                np.hypot(x - kx, y - ky) < 25 for (kx, ky, kr) in deduped
            )
            if not too_close:
                deduped.append((x, y, r))

        # --------------------------------------------------------------
        # NEW: जास्तीत जास्त बटणं मर्यादित करा (सामान्य शर्टला 5-8 बटणं असतात)
        # Hough आधीच confidence नुसार sorted देतो, त्यामुळे top-N घ्या
        # --------------------------------------------------------------
        deduped = deduped[:max_buttons]

        return deduped

    #########################################################################

    def draw_synthetic_button(self, image, x, y, r):
        overlay = image.copy()

        # स्टार्क पांढऱ्याऐवजी आजूबाजूच्या fabric रंगावर आधारित हलका रंग
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

        # फक्त 2 सूक्ष्म, low-contrast थ्रेड-होल्स (4 नाही -> football pattern टाळण्यासाठी)
        hole_r = max(1, r // 6)
        cv2.circle(overlay, (x - r // 4, y), hole_r, (120, 120, 120), -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x + r // 4, y), hole_r, (120, 120, 120), -1, lineType=cv2.LINE_AA)

        # कमी opacity -> बटण फॅब्रिकवर हळुवारपणे बसतं, ठळक/कापल्यासारखं दिसत नाही
        cv2.addWeighted(overlay, 0.82, image, 0.18, 0, dst=image)
        return image

    ###########################################
    # pocket
    ##############################################
    def detect_pocket_outline(self, person_image, shirt_mask):
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

            # NEW: खूप मोठा minimum-area threshold -> fabric texture noise आपोआप गळून पडतो
            if area < 0.015 * shirt_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h + 1e-6)
            if not (0.7 < aspect < 2.2):
                continue

            rel_y = (y - ys.min()) / float(shirt_h)
            if not (0.18 < rel_y < 0.45):
                continue

            # NEW: आकार साधारण आयताकृती (चौकोनी) आहे का ते तपासा
            approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
            if not (4 <= len(approx) <= 8):
                continue

            # NEW: भरीव, घन आकार हवा -- विस्कळीत, तुटक रेषा नको
            fill_ratio = area / float(w * h + 1e-6)
            if fill_ratio < 0.55:
                continue

            # NEW: फक्त सर्वात मोठा, सर्वात विश्वासार्ह उमेदवार ठेवा (एकच pocket असतो)
            if area > best_area:
                best_area = area
                best_candidate = c

        return [best_candidate] if best_candidate is not None else []

    def draw_pocket_outline(self, image, contours):
        """
        detect_pocket_outline ने शोधलेल्या contour ला
        इमेजवर एक हलकी, पातळ stitch-line म्हणून रेखाटतो.
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

        NOTE
        ----
        This milestone does NOT preserve lighting or folds.
        That will be implemented in the next milestone.
        """

        # --------------------------------------------------------------
        # Convert the SAM mask into a binary mask.
        #
        # Background = 0
        # Shirt = 255
        # --------------------------------------------------------------
        binary_mask = (shirt_mask > 0).astype("uint8") * 255

        # --------------------------------------------------------------
        # Convert the single-channel mask into a 3-channel mask.
        #
        # Person Image : (H, W, 3)
        # Fabric Image : (H, W, 3)
        # Mask         : (H, W)
        #
        # We need:
        # Mask         : (H, W, 3)
        # --------------------------------------------------------------
        binary_mask = cv2.merge([
            binary_mask,
            binary_mask,
            binary_mask
        ])

        # --------------------------------------------------------------
        # Copy the original image.
        #
        # We never modify the original image directly.
        # --------------------------------------------------------------
        output = person_image.copy()

        # --------------------------------------------------------------
        # Wherever the shirt mask is white,
        # replace those pixels with the prepared fabric.
        # --------------------------------------------------------------
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
        1. Validate inputs.
        2. Prepare fabric.
        3. Preserve original lighting.
        4. Apply shirt mask.
        """
        # ----------------------------------------------------------
        # Extract analyzed fabric information.
        # ----------------------------------------------------------

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


        # pattern_repeat = fabric_info["pattern_repeat"]

        print("Before prepare:", fabric_image.shape)
        print("Fabric dtype:", fabric_image.dtype)

        # Step 1
        # prepared_fabric = self.prepare_fabric(
        #     fabric_image=fabric_image,
        #     target_width=person_image.shape[1],
        #     target_height=person_image.shape[0],
        #     repeat_size=pattern_repeat
        # )

        prepared_fabric=fabric_image

        print("After prepare:", prepared_fabric.shape)
        print("prepared dtype:", prepared_fabric.dtype)

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug_1_prepared_fabric2.png"),
            prepared_fabric
        )
        # Step 2
        # ----------------------------------------------------------
        # Preserve overall lighting.
        # ----------------------------------------------------------

        # ----------------------------------------------------------
        # NEW: preserve_lighting चा वेगळा multiply काढला —
        # RTV shading_map आधीच illumination + folds दोन्ही सांभाळतो.
        # दोन्ही एकत्र वापरल्याने brightness दुप्पट होत होती.
        # ----------------------------------------------------------
        realistic_fabric = prepared_fabric

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug_2_after_lighting2.png"),
            realistic_fabric
        )

        # ----------------------------------------------------------
        # Extract fine details from the original shirt.
        # ----------------------------------------------------------

        # ----------------------------------------------------------
        # Extract shirt structure.
        # ----------------------------------------------------------

        # fold_map = self.extract_fold_map(
        #     person_image,
        #     shirt_mask
        # )
        #
        # fold_map = self.clean_fold_map(
        #     fold_map
        # )
        #
        # cv2.imwrite(
        #     str(DEBUG_FOLDER / "debug_3_after_fold_map2.png"),
        #     fold_map
        # )
        #
        # realistic_fabric = self.apply_fold_map(
        #     realistic_fabric,
        #     fold_map,
        #     strength=0.18
        # )
        #
        # cv2.imwrite(
        #     str(DEBUG_FOLDER / "debug_4_after_realastic_fabric2.png"),
        #     realistic_fabric
        # )

        pattern_repeat = fabric_info.get("pattern_repeat")

        # ----------------------------------------------------------
        # NEW: शर्टाची प्रिंट किती busy आहे ते मोजा
        # ----------------------------------------------------------
        busyness = self.estimate_shirt_busyness(
            person_image,
            shirt_mask
        )

        print("Shirt busyness score:", busyness)

        shading_map = self.extract_structure_map_rtv(
            person_image,
            shirt_mask,
            pattern_repeat=pattern_repeat,
            busyness=busyness
        )

        # ----------------------------------------------------------
        # NEW: busy प्रिंट असेल तर shading map चा प्रभाव कमी करा
        # (1.0 = neutral, कोणताही effect नाही)
        # ----------------------------------------------------------
        shading_map = self.separate_real_folds_from_texture(
            shading_map,
            busyness=busyness,
            large_fold_radius=25
        )

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

        realistic_fabric = self.apply_structure_map_lab(
            realistic_fabric,
            shading_map
        )
        cv2.imwrite(
            str(DEBUG_FOLDER / "debug_4_after_realistic_fabric2.png"),
            realistic_fabric
        )

        # ----------------------------------------------------------
        # Replace only the shirt region.
        # ----------------------------------------------------------

        output = self.apply_shirt_mask(
            person_image,
            shirt_mask,
            realistic_fabric
        )

        # ----------------------------------------------------------
        # NEW: Buttons आणि Pocket outline जोडा
        # ----------------------------------------------------------
        buttons = self.detect_buttons(person_image, shirt_mask)
        print("Buttons found:", len(buttons))

        for (bx, by, br) in buttons:
            output = self.draw_synthetic_button(output, bx, by, br)

        pocket_contours = self.detect_pocket_outline(person_image, shirt_mask)
        print("Pocket candidates found:", len(pocket_contours))

        output = self.draw_pocket_outline(output, pocket_contours)

        return output

