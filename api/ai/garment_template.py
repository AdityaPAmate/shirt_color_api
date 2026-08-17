"""
File:
    api/ai/garment_template.py

Purpose:
    Convert a SAM shirt mask into a simple rule-based Kurta mask.

    The original shirt mask is preserved. Any lower connected artifact is
    removed first. The Kurta extension then continues naturally from the
    real lower shirt hem.

Geometry:
    - Kurta height: extend_height_ratio * real shirt height.
    - The real shirt bottom is detected from the widest central torso row
      in the lower body region; a lower artifact (if any) is ignored and
      then physically removed from the mask.
    - Extension starts from the real shirt bottom, using the actual
      bottom torso width.
    - Width increases by up to max_side_growth_ratio * base_width on
      EACH side, from top to bottom of the extension, using an ease-in
      (quadratic) curve so the flare is subtle near the shirt hem and
      more pronounced near the bottom -> looks like real Kurta drape
      instead of a near-rectangle.
    - Bottom corners are rounded.

CHANGELOG
---------
NEW: max_side_growth is now a RATIO of the base (hem) width, not a fixed
     pixel count. A fixed pixel value (e.g. 3px) is invisible on a
     realistic-resolution photo, which is why the previous version's
     Kurta looked almost the same width as the shirt instead of flaring
     outward. Scaling by base width keeps the flare proportionate at
     any image resolution.
NEW: Growth now follows an ease-in (quadratic) curve instead of linear,
     so the flare is gentle just below the hem and increases more
     noticeably toward the bottom -> closer to how a real Kurta drapes.
NEW: extend_height_ratio default raised to 0.45 (from 0.15/0.25) for a
     proportionate Kurta length relative to the shirt.
"""

import cv2
import numpy as np


class GarmentTemplate:

    def __init__(self):
        pass

    def _largest_component(self, mask_bin):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_bin, connectivity=8
        )

        if num_labels <= 1:
            return mask_bin

        largest_label = 1 + int(
            np.argmax(stats[1:, cv2.CC_STAT_AREA])
        )
        return (labels == largest_label).astype(np.uint8)

    @staticmethod
    def _row_runs(row):
        xs = np.where(row > 0)[0]

        if len(xs) == 0:
            return []

        breaks = np.where(np.diff(xs) > 1)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(xs) - 1]

        return [
            (int(xs[s]), int(xs[e]))
            for s, e in zip(starts, ends)
        ]

    def _central_run(self, mask_bin, y, center_x):
        if y < 0 or y >= mask_bin.shape[0]:
            return None

        runs = self._row_runs(mask_bin[y])
        print('Runs are: ', runs)
        print()

        if not runs:
            return None

        # torso साठी image center मध्ये असलेला run प्राधान्याने निवडतो.
        containing = [
            r for r in runs
            if r[0] <= center_x <= r[1]
        ]

        if containing:
            return max(
                containing,
                key=lambda r: r[1] - r[0] + 1
            )

        # Fallback: center च्या सर्वात जवळचा run.
        return min(
            runs,
            key=lambda r: abs(
                ((r[0] + r[1]) / 2.0) - center_x
            )
        )

    def _find_real_bottom(self, mask_bin, y_top, center_x):
        """
        Find the real shirt hem.

        The unwanted lower object (if any) may be connected to the shirt,
        so merely taking max(y) is wrong. The real lower torso reaches its
        maximum central width at the shirt hem; any unwanted lower
        extension becomes progressively narrower after that point.
        """
        h = mask_bin.shape[0]
        shirt_h = h - y_top

        search_start = int(round(y_top + shirt_h * 0.48))
        search_end = h - 1

        profile = []

        for y in range(search_start, search_end + 1):
            run = self._central_run(mask_bin, y, center_x)
            if run is None:
                continue

            width = run[1] - run[0] + 1

            if width >= 30:
                profile.append((y, width, run[0], run[1]))

        if not profile:
            ys = np.where(mask_bin > 0)[0]
            return int(ys.max())

        widths = np.array([p[1] for p in profile], dtype=np.float32)

        kernel = 11
        if len(widths) >= kernel:
            smooth = np.convolve(
                widths,
                np.ones(kernel) / kernel,
                mode="same"
            )
            margin = kernel // 2
            smooth[:margin] = widths[:margin]
            smooth[-margin:] = widths[-margin:]
        else:
            smooth = widths

        peak_index = int(np.argmax(smooth))

        peak_value = smooth[peak_index]
        near_peak = np.where(
            smooth >= peak_value * 0.995
        )[0]

        local = near_peak[
            (near_peak >= max(0, peak_index - 30)) &
            (near_peak <= min(len(profile) - 1, peak_index + 30))
        ]

        chosen_index = int(
            round(float(np.mean(local)))
        ) if len(local) else peak_index

        return int(profile[chosen_index][0])

    @staticmethod
    def _bottom_rounding(left, right, progress, radius):
        """
        Round only the bottom corners.
        """
        width = right - left + 1
        radius = min(radius, width // 2)

        if radius <= 1:
            return left, right

        p = float(np.clip(progress, 0.0, 1.0))
        d = radius * p

        shrink = radius - np.sqrt(
            max(0.0, radius * radius - d * d)
        )

        new_left = int(round(left + shrink))
        new_right = int(round(right - shrink))

        return new_left, new_right

    def shirt_to_kurta_mask(
        self,
        shirt_mask,
        extend_height_ratio=0.40,
        start_offset_ratio=0.05,
        max_side_growth_ratio=0.11,
        bottom_curve_ratio=0.12,
    ):
        """
        Convert shirt mask -> Kurta mask.

        extend_height_ratio:
            Kurta साठी shirt height च्या तुलनेत किती extra height
            (0.45 = 45% जास्त लांबी).

        start_offset_ratio:
            Hem च्या किंचित वर reference width काढण्यासाठी offset.

        max_side_growth_ratio:
            तळाशी, base (hem) width च्या तुलनेत प्रत्येक बाजूला
            जास्तीत जास्त किती वाढ (0.35 = base width च्या 35% इतकी
            जास्त रुंदी प्रत्येक बाजूला, तळाशी).

        bottom_curve_ratio:
            नवीन extension पैकी किती भाग rounded-corner साठी वापरायचा.
        """
        original_dtype = shirt_mask.dtype

        mask_bin = (shirt_mask > 0).astype(np.uint8)
        mask_bin = self._largest_component(mask_bin)
        """<----- here largest connected component returned and now mask bin contain cleaned/main shirt component """

        ys = np.where(mask_bin > 0)[0]

        if len(ys) == 0:
            return shirt_mask

        img_height, img_width = mask_bin.shape
        y_top = int(ys.min())
        center_x = (img_width - 1) / 2.0

        # ---------------------------------------------------------
        # 1. खरा shirt bottom (hem) शोध.
        # ---------------------------------------------------------
        y_bottom = self._find_real_bottom(mask_bin, y_top, center_x)

        shirt_height = max(1, y_bottom - y_top)

        # ---------------------------------------------------------
        # 2. खऱ्या hem च्या खालचं सगळं (जुना artifact असेल तर तोही)
        # पूर्णपणे पुसून टाक.
        # ---------------------------------------------------------
        clean_mask = mask_bin.copy()
        clean_mask[y_bottom + 1:, :] = 0

        # ---------------------------------------------------------
        # 3. Hem च्या किंचित वरचा reference row.
        # ---------------------------------------------------------
        offset_px = max(2, int(round(shirt_height * start_offset_ratio)))
        reference_y = max(y_top, y_bottom - offset_px)

        reference_run = self._central_run(
            clean_mask, reference_y, center_x
        )

        if reference_run is None:
            reference_run = self._central_run(
                clean_mask, y_bottom, center_x
            )

        if reference_run is None:
            return shirt_mask

        # ---------------------------------------------------------
        # 4. खरी hem width (काही rows चा median घेऊन स्थिर करणे).
        # -------------------x--------------------------------------
        base_left = reference_run[0]
        base_right = reference_run[1]

        nearby_left = []
        nearby_right = []

        band = max(3, int(round(shirt_height * 0.015)))

        for y in range(
            max(y_top, reference_y - band),
            min(y_bottom, reference_y + band) + 1
        ):
            run = self._central_run(clean_mask, y, center_x)
            if run is not None:
                nearby_left.append(run[0])
                nearby_right.append(run[1])

        if nearby_left:
            base_left = int(round(np.median(nearby_left)))
            base_right = int(round(np.median(nearby_right)))

        base_width = max(1, base_right - base_left)

        # ---------------------------------------------------------
        # 5. Extension height.
        # ---------------------------------------------------------
        extend_px = max(1, int(round(shirt_height * extend_height_ratio)))
        new_bottom = min(img_height - 1, y_bottom + extend_px)
        # ---------------------------------------------------------
        # 6. FIXED: growth आता base_width च्या ratio नुसार, pixels
        # मध्ये convert करून घेतो -> कुठल्याही resolution वर
        # प्रमाणबद्ध flare दिसेल.
        # ---------------------------------------------------------
        max_side_growth_px = base_width * max_side_growth_ratio

        kurta_mask = clean_mask.copy()

        extension_height = max(1, new_bottom - y_bottom)

        curve_height = max(4, int(round(extension_height * bottom_curve_ratio)))
        curve_start = max(y_bottom + 1, new_bottom - curve_height + 1)

        for y in range(y_bottom + 1, new_bottom + 1):

            p = (y - y_bottom) / extension_height

            # NEW: ease-in (quadratic) curve -> hem जवळ किंचित flare,
            # तळाकडे जास्त flare (खऱ्या Kurta च्या drape सारखं).
            eased_p = p * p

            growth = max_side_growth_px * eased_p

            left = int(round(base_left - growth))
            right = int(round(base_right + growth))

            if y >= curve_start:
                curve_p = (y - curve_start) / max(1, new_bottom - curve_start)
                radius = max(2, int(round(curve_height)))

                left, right = self._bottom_rounding(
                    left, right, curve_p, radius
                )

            left = max(0, left)
            right = min(img_width - 1, right)

            if left <= right:
                kurta_mask[y, left:right + 1] = 1

        # ---------------------------------------------------------
        # 7. Original dtype परत कर.
        # ---------------------------------------------------------
        if original_dtype == bool:
            return kurta_mask.astype(bool)

        return (kurta_mask * 255).astype(original_dtype)

    def get_garment_mask(self, shirt_mask, garment_type="shirt"):
        if garment_type.lower() == "kurta":
            return self.shirt_to_kurta_mask(
                shirt_mask,
                extend_height_ratio=0.40,
                start_offset_ratio=0.05,
                max_side_growth_ratio=0.11,
                bottom_curve_ratio=0.12,
            )

        return shirt_mask