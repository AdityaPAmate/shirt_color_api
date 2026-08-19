"""
File:
    api/ai/garment_template.py

Purpose:
    Convert a SAM shirt mask into a simple rule-based Kurta mask.

Updated bottom-detection logic:
    1. Convert SAM mask into binary mask.
    2. Find the actual top and actual bottom of the original mask.
    3. Use the lower 40% of the actual mask as the bottom-search region.
    4. Search from the bottom upward instead of searching for the maximum width.
    5. Record row-by-row geometry:
           - left boundary
           - right boundary
           - width
           - left zero count
           - right zero count
           - total zero count
    6. Detect a possible inverted-V (∧) bottom cut from the row pattern.
    7. Determine the real shirt hem (y_bottom).
    8. Remove the unwanted region below the real hem.
    9. Restore the detected inverted-V notch when it belongs to the shirt body.
   10. Calculate Kurta extension height from the ACTUAL ORIGINAL MASK HEIGHT.
   11. Extend the mask downward from the detected real hem.
   12. Apply proportional side growth using the hem/base width.
   13. Round the bottom corners.

Important:
    The extension height is NOT calculated from:
        y_bottom - y_top

    Instead:
        actual_mask_height = original_bottom_y - y_top
        extension_height = actual_mask_height * extend_height_ratio
"""

import cv2
import numpy as np


class GarmentTemplate:

    def __init__(self):
        pass

    # ================================================================
    # 1. KEEP ONLY THE LARGEST CONNECTED MASK COMPONENT
    # ================================================================
    def _largest_component(self, mask_bin):
        """
        Keep only the largest connected white component.

        Input:
            mask_bin:
                Binary mask containing 0 and 1.

        Output:
            Binary mask containing only the largest connected component.
        """

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_bin,
            connectivity=8
        )

        if num_labels <= 1:
            return mask_bin

        largest_label = 1 + int(
            np.argmax(stats[1:, cv2.CC_STAT_AREA])
        )

        return (labels == largest_label).astype(np.uint8)

    # ================================================================
    # 2. FIND CONTINUOUS WHITE REGIONS IN ONE ROW
    # ================================================================
    @staticmethod
    def _row_runs(row):
        """
        Find continuous white pixel runs in one horizontal row.

        Example:

            0 0 1 1 1 1 0 0

        Result:

            [(2, 5)]

        Each tuple contains:
            (start_x, end_x)
        """

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

    # ================================================================
    # 3. FIND THE CENTRAL WHITE RUN
    # ================================================================
    def _central_run(self, mask_bin, y, center_x):
        """
        Return the white run that contains the image center.

        If the center is not inside any white run, return the run
        whose center is closest to the image center.
        """

        if y < 0 or y >= mask_bin.shape[0]:
            return None

        runs = self._row_runs(mask_bin[y])

        if not runs:
            return None

        containing = [
            r for r in runs
            if r[0] <= center_x <= r[1]
        ]

        if containing:
            return max(
                containing,
                key=lambda r: r[1] - r[0] + 1
            )

        return min(
            runs,
            key=lambda r: abs(
                ((r[0] + r[1]) / 2.0) - center_x
            )
        )

    # ================================================================
    # 4. RECORD GEOMETRY OF ONE ROW
    # ================================================================
    def _analyze_row(self, mask_bin, y, center_x):
        """
        Analyze one horizontal row.

        CHANGED: now also records `run_count` -- how many separate
        white runs exist in this row (not just the central one).
        This is essential for the new hem-detection logic: a row
        with run_count > 1 means the garment has split into two
        pieces at this height (a V-notch), and must be treated as
        INVALID for hem detection, not just measured by its central
        run's width as before.
        """

        height, width = mask_bin.shape

        all_runs = self._row_runs(mask_bin[y])
        run_count = len(all_runs)

        run = self._central_run(mask_bin, y, center_x)

        if run is None:
            return {
                "y": int(y),
                "left": None,
                "right": None,
                "width": 0,
                "left_zero": 0,
                "right_zero": 0,
                "total_zero": width,
                "run_count": int(run_count)
            }

        left, right = run

        left_zero = left
        right_zero = width - 1 - right
        total_zero = left_zero + right_zero

        row_info = {
            "y": int(y),
            "left": int(left),
            "right": int(right),
            "width": int(right - left + 1),
            "left_zero": int(left_zero),
            "right_zero": int(right_zero),
            "total_zero": int(total_zero),
            "run_count": int(run_count)
        }

        return row_info
    # ================================================================
    # 5. FIND ACTUAL ORIGINAL MASK BOUNDS
    # ================================================================
    def _get_actual_mask_bounds(self, mask_bin):
        """
        Find the actual top-most and bottom-most white pixels
        from the original binary mask.

        This is used ONLY for calculating the actual mask height.

        Important:
            This bottom is NOT the final real shirt hem.

        It is simply the bottom-most white pixel present in the
        original SAM mask.
        """

        ys = np.where(mask_bin > 0)[0]

        if len(ys) == 0:
            return None, None, 0

        y_top = int(ys.min())
        actual_bottom = int(ys.max())

        # Actual original mask height.
        actual_mask_height = max(
            1,
            actual_bottom - y_top
        )

        return y_top, actual_bottom, actual_mask_height

    # ================================================================
    # 6. ANALYZE LOWER 40% FROM BOTTOM TO TOP
    # ================================================================
    def _analyze_lower_region(
        self,
        mask_bin,
        y_top,
        actual_bottom,
        lower_region_ratio=0.40
    ):
        """
        Analyze the lower 40% of the original mask.

        Search direction:
            bottom -> upward

        This is intentionally different from the old logic.

        The old logic searched for the row having maximum central
        width.

        The new logic starts from the actual bottom because the
        garment physically ends at the bottom side of the mask.

        Every row is recorded for later hem / inverted-V analysis.
        """

        height, width = mask_bin.shape

        actual_mask_height = max(
            1,
            actual_bottom - y_top
        )

        # Start of the lower 40% region.
        lower_region_start = int(
            round(
                actual_bottom -
                actual_mask_height * lower_region_ratio
            )
        )

        lower_region_start = max(
            y_top,
            lower_region_start
        )

        center_x = (width - 1) / 2.0

        row_data = []

        # ------------------------------------------------------------
        # Search from bottom upward.
        # ------------------------------------------------------------
        for y in range(
            actual_bottom,
            lower_region_start - 1,
            -1
        ):

            info = self._analyze_row(
                mask_bin,
                y,
                center_x
            )

            row_data.append(info)
        print(row_data)

        return row_data, lower_region_start

    # ================================================================
    # 7. DETECT INVERTED-V (∧) BOTTOM CUT
    # ================================================================
    def _detect_inverted_v_cut(
        self,
        row_data,
        min_zero_change=1,
        min_consecutive_rows=2
    ):
        """
        Detect a possible inverted-V (∧) bottom cut.

        Important geometry:

            In an inverted-V cut:

                    upper center
                         /\
                        /  \
                       /    \
                      /      \

            In image coordinates, the center portion of the hem
            appears higher while the side portions continue lower.

        Therefore, while moving upward from the bottom, we may observe
        the garment width changing in a structured way rather than
        simply seeing random segmentation noise.

        This function does NOT immediately modify the mask.

        It only records and returns the candidate rows so that the
        final y_bottom decision can use this information.
        """

        if len(row_data) < min_consecutive_rows + 1:
            return []

        candidates = []

        # ------------------------------------------------------------
        # Compare neighbouring rows.
        #
        # Because row_data is stored bottom -> upward, index i-1 is
        # physically below index i.
        # ------------------------------------------------------------
        for i in range(1, len(row_data)):

            current = row_data[i]
            previous = row_data[i - 1]

            if (
                current["left"] is None or
                previous["left"] is None
            ):
                continue

            left_change = (
                current["left"] -
                previous["left"]
            )

            right_change = (
                previous["right"] -
                current["right"]
            )

            # A structured change on both sides is more meaningful
            # than a change on only one side.
            if (
                left_change >= min_zero_change and
                right_change >= min_zero_change
            ):
                candidates.append(current)

        # ------------------------------------------------------------
        # Keep only meaningful consecutive candidate rows.
        # ------------------------------------------------------------
        if not candidates:
            return []

        grouped = []
        current_group = [candidates[0]]

        for item in candidates[1:]:

            previous_y = current_group[-1]["y"]

            if abs(item["y"] - previous_y) <= 1:
                current_group.append(item)
            else:
                if len(current_group) >= min_consecutive_rows:
                    grouped.append(current_group)

                current_group = [item]

        if len(current_group) >= min_consecutive_rows:
            grouped.append(current_group)

        # Return the largest continuous candidate group.
        if not grouped:
            return []

        return max(
            grouped,
            key=lambda group: len(group)
        )

    # ================================================================
    # 8. FIND REAL SHIRT BOTTOM
    # ================================================================
    def _find_real_bottom(
        self,
        mask_bin,
        y_top,
        actual_bottom,
        lower_region_ratio=0.40,
        stability_window=8,
        width_stability_ratio=0.65,
        width_variation_tolerance=0.20
    ):
        """
        Find the real shirt hem using bottom-up STABILITY analysis.

        REPLACED APPROACH (per user's explicit correction, verified
        against an actual failing mask):
        ------------------------------------------------------------
        The previous version detected an inverted-V (^) notch and
        used its narrowest tip to measure base_width -- this
        produced a thin, wrong "pillar" Kurta extension. Verified
        with actual row data from the failing mask: the V-tip region
        was only 17-65px wide, while the true torso width nearby was
        150-245px.

        The user manually marked the correct hem on a real mask
        (a roughly horizontal line) and clarified:
            - The V-notch / narrow tail below it must be DISCARDED,
              not restored.
            - The true hem is the last row (scanning bottom-up) that
              is still part of a STABLE, WIDE, single-piece garment
              row -- i.e. right before the mask starts narrowing or
              splitting into the V/tail shape.

        New algorithm:
            1. Scan the lower `lower_region_ratio` of the actual
               mask, bottom -> upward, recording for every row:
                   - run_count (how many separate white pieces)
                   - width of the central run
            2. Compute max_width_in_region = widest row seen here.
               This is the size reference: a real hem row should not
               be drastically narrower than the widest nearby point.
            3. A row is VALID only if:
                   a. run_count == 1        (not split by a V-notch)
                   b. width >= max_width_in_region * width_stability_ratio
                      (not a narrow tip/tail/artifact)
            4. Find the first (bottom-most) group of
               `stability_window` CONSECUTIVE valid rows whose
               widths do not vary from each other by more than
               `width_variation_tolerance` (relative to the group's
               median width). Requiring several consecutive rows
               (not just one) avoids a single lucky-width noisy row
               being mistaken for the real hem.
            5. y_bottom = the bottom-most (largest y) row of that
               first stable group. Everything below it is discarded.
               The V-notch / narrow tail is NOT restored.

        Verified on an actual failing mask: this logic detected
        y_bottom = 397, while the user's own hand-marked reference
        line sat at y = 382-388 -- a ~10px difference, consistent
        with the line being hand-drawn and approximate, and with
        y=397 sitting right at the boundary where the real V-notch
        begins (observed at y=399 in that mask).

        Returns:
            y_bottom : int, Y-coordinate of the detected real hem.
            row_data : list of per-row records (for debugging).
            None     : kept as 3rd return value for compatibility
                       with the calling code -- V-cut restoration is
                       no longer used, so there is no v_cut_rows
                       payload anymore.
        """

        row_data, lower_start = self._analyze_lower_region(
            mask_bin,
            y_top,
            actual_bottom,
            lower_region_ratio
        )

        if not row_data:
            return actual_bottom, [], None

        # ------------------------------------------------------------
        # Reference width: widest row seen anywhere in the scanned
        # lower region. A real hem/torso row should stay close to
        # this, not be drastically narrower.
        # ------------------------------------------------------------
        widths_in_region = [
            info["width"] for info in row_data if info["width"] > 0
        ]

        if not widths_in_region:
            return actual_bottom, row_data, None

        max_width_in_region = max(widths_in_region)
        min_valid_width = max_width_in_region * width_stability_ratio

        # ------------------------------------------------------------
        # Mark each row VALID / INVALID using the two measurable
        # rules described above.
        # ------------------------------------------------------------
        for info in row_data:
            info["valid"] = (
                info["run_count"] == 1 and
                info["width"] >= min_valid_width
            )

        # ------------------------------------------------------------
        # row_data is already ordered bottom -> upward. Find the
        # first run of `stability_window` consecutive VALID rows
        # whose widths stay close to each other.
        # ------------------------------------------------------------
        n = len(row_data)

        for start_idx in range(0, n - stability_window + 1):

            window = row_data[start_idx:start_idx + stability_window]

            if not all(item["valid"] for item in window):
                continue

            window_widths = [item["width"] for item in window]
            median_width = float(np.median(window_widths))

            if median_width <= 0:
                continue

            max_dev = max(
                abs(w - median_width) for w in window_widths
            )

            if (max_dev / median_width) <= width_variation_tolerance:
                # Bottom-most row of this stable window = real hem.
                y_bottom = window[0]["y"]
                return int(y_bottom), row_data, None

        # ------------------------------------------------------------
        # Fallback: no stable window found in the lower region.
        # This should be rare -- flagged clearly rather than
        # silently guessing further.
        # ------------------------------------------------------------
        fallback_y = row_data[-1]["y"]
        return int(fallback_y), row_data, None

    # ================================================================
    # 9. RESTORE INVERTED-V REGION
    # ================================================================
    def _restore_inverted_v_region(
        self,
        mask_bin,
        y_bottom,
        v_cut_rows
    ):
        """
        Restore the black area belonging to the inverted-V cut.

        Why this is needed:

            After y_bottom is selected, everything below it may have
            been cleared.

            If the original shirt has an inverted-V bottom cut,
            some pixels that belong to the shirt body can remain black
            above the detected bottom line.

            Those pixels must be restored before creating the
            extension.

        The restoration is conservative:
            - only the detected V candidate rows are considered.
            - the central region is restored based on neighbouring
              garment boundaries.
        """

        if not v_cut_rows:
            return mask_bin

        restored_mask = mask_bin.copy()

        height, width = restored_mask.shape

        # ------------------------------------------------------------
        # Determine the widest valid garment boundaries observed in
        # the V candidate region.
        # ------------------------------------------------------------
        valid_rows = [
            item for item in v_cut_rows
            if (
                item["left"] is not None and
                item["right"] is not None
            )
        ]

        if not valid_rows:
            return restored_mask

        min_left = min(
            item["left"]
            for item in valid_rows
        )

        max_right = max(
            item["right"]
            for item in valid_rows
        )

        min_left = max(
            0,
            min_left
        )

        max_right = min(
            width - 1,
            max_right
        )

        # ------------------------------------------------------------
        # Restore the central V-related region only above y_bottom.
        # Do not modify the already removed lower region.
        # ------------------------------------------------------------
        for item in valid_rows:

            y = item["y"]

            if y > y_bottom:
                continue

            left = item["left"]
            right = item["right"]

            if left is None or right is None:
                continue

            left = max(0, left)
            right = min(width - 1, right)

            if left <= right:
                restored_mask[
                    y,
                    left:right + 1
                ] = 1

        return restored_mask

    # ================================================================
    # 10. FIND STABLE BASE WIDTH NEAR REAL HEM
    # ================================================================
    def _find_base_width(
        self,
        clean_mask,
        y_top,
        y_bottom,
        shirt_height,
        start_offset_ratio=0.05
    ):
        """
        Find the base width near the real shirt hem.

        REVERTED to the simple median-based single-run approach.
        The earlier v_cut-envelope special case is no longer needed:
        the new `_find_real_bottom()` already guarantees y_bottom
        lands on a stable, wide, single-piece row -- it never lands
        inside a V-notch or narrow tail anymore, so no special
        handling is required here.
        """

        height, width = clean_mask.shape
        center_x = (width - 1) / 2.0

        offset_px = max(
            2,
            int(round(shirt_height * start_offset_ratio))
        )

        reference_y = max(y_top, y_bottom - offset_px)

        reference_run = self._central_run(clean_mask, reference_y, center_x)

        if reference_run is None:
            reference_run = self._central_run(clean_mask, y_bottom, center_x)

        if reference_run is None:
            return None

        base_left = reference_run[0]
        base_right = reference_run[1]

        band = max(3, int(round(shirt_height * 0.015)))

        nearby_left = []
        nearby_right = []

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

        base_width = max(1, base_right - base_left + 1)

        return {
            "left": int(base_left),
            "right": int(base_right),
            "width": int(base_width),
            "reference_y": int(reference_y)
        }
    # ================================================================
    # 11. ROUND BOTTOM CORNERS
    # ================================================================
    @staticmethod
    def _bottom_rounding(
        left,
        right,
        progress,
        radius
    ):
        """
        Round only the bottom corners.
        """

        width = right - left + 1

        radius = min(
            radius,
            width // 2
        )

        if radius <= 1:
            return left, right

        p = float(
            np.clip(
                progress,
                0.0,
                1.0
            )
        )

        d = radius * p

        shrink = radius - np.sqrt(
            max(
                0.0,
                radius * radius - d * d
            )
        )

        new_left = int(
            round(
                left + shrink
            )
        )

        new_right = int(
            round(
                right - shrink
            )
        )

        return new_left, new_right

    # ================================================================
    # 12. MAIN SHIRT -> KURTA MASK CONVERSION
    # ================================================================
    def shirt_to_kurta_mask(
        self,
        shirt_mask,
        extend_height_ratio=0.40,
        start_offset_ratio=0.05,
        max_side_growth_ratio=0.11,
        bottom_curve_ratio=0.12
    ):
        """
        Convert a shirt mask into a Kurta mask.

        Main geometry:

            Actual original mask height:
                actual_bottom - y_top

            Extension height:
                actual_mask_height * 0.40

            Extension starts from:
                detected real y_bottom

            Extension base width:
                stable width near detected hem

            Side growth:
                base_width * max_side_growth_ratio

        """

        # ------------------------------------------------------------
        # STAGE 1
        # Convert the incoming SAM mask into a clean binary mask.
        #
        # Data changes:
        #   Original SAM mask
        #          ↓
        #   Binary 0 / 1 mask
        # ------------------------------------------------------------
        original_dtype = shirt_mask.dtype

        mask_bin = (
            shirt_mask > 0
        ).astype(np.uint8)

        # ------------------------------------------------------------
        # STAGE 2
        # Remove disconnected small components.
        #
        # Data changes:
        #   Binary mask
        #          ↓
        #   Largest connected shirt component
        # ------------------------------------------------------------
        mask_bin = self._largest_component(
            mask_bin
        )

        # ------------------------------------------------------------
        # STAGE 3
        # Find actual original mask boundaries.
        #
        # IMPORTANT:
        # actual_bottom is only the bottom-most white pixel of the
        # original mask.
        #
        # It is used for ACTUAL MASK HEIGHT calculation.
        #
        # It is NOT automatically accepted as the real shirt hem.
        # ------------------------------------------------------------
        y_top, actual_bottom, actual_mask_height = (
            self._get_actual_mask_bounds(
                mask_bin
            )
        )

        if y_top is None:
            return shirt_mask

        img_height, img_width = mask_bin.shape

        # ------------------------------------------------------------
        # STAGE 4
        # Find the real shirt bottom using the NEW bottom-up logic.
        #
        # Search is restricted to the lower 40% of the original mask.
        # ------------------------------------------------------------
        y_bottom, row_data, v_cut_rows = (
            self._find_real_bottom(
                mask_bin,
                y_top,
                actual_bottom,
                lower_region_ratio=0.40
            )
        )

        # ------------------------------------------------------------
        # STAGE 5
        # Create a temporary mask where everything below the detected
        # real shirt hem is removed.
        #
        # Data changes:
        #   Original binary mask
        #          ↓
        #   Clean shirt mask
        # ------------------------------------------------------------
        clean_mask = mask_bin.copy()

        clean_mask[
            y_bottom + 1:,
            :
        ] = 0

        # ------------------------------------------------------------
        # STAGE 6 -- INTENTIONALLY SKIPPED (per explicit correction):
        # _find_real_bottom() no longer lands inside a V-notch, so
        # there is nothing to restore. The V-notch/narrow-tail region
        # is discarded, not restored. _restore_inverted_v_region() is
        # kept defined in this file but is no longer called.
        # ------------------------------------------------------------
        # clean_mask = self._restore_inverted_v_region(
        #     clean_mask,
        #     y_bottom,
        #     v_cut_rows
        # )

        # ------------------------------------------------------------
        # STAGE 7
        # Find the base width near the real shirt hem.
        #
        # IMPORTANT:
        # base_width is NOT the width of the black region.
        #
        # It comes from the central white garment run close to
        # y_bottom.
        # ------------------------------------------------------------
        base_info = self._find_base_width(
            clean_mask,
            y_top,
            y_bottom,
            actual_mask_height,
            start_offset_ratio=start_offset_ratio
        )

        if base_info is None:
            return shirt_mask

        base_left = base_info["left"]
        base_right = base_info["right"]
        base_width = base_info["width"]

        # ------------------------------------------------------------
        # STAGE 8
        # Calculate extension height.
        #
        # IMPORTANT CHANGE:
        #
        # OLD:
        #     y_bottom - y_top
        #
        # NEW:
        #     actual original mask height
        #
        #     actual_mask_height * 0.40
        #
        # Therefore the unwanted lower artifact does not define the
        # extension height through y_bottom.
        # ------------------------------------------------------------
        extend_px = max(
            1,
            int(
                round(
                    actual_mask_height *
                    extend_height_ratio
                )
            )
        )

        new_bottom = min(
            img_height - 1,
            y_bottom + extend_px
        )

        # ------------------------------------------------------------
        # STAGE 9
        # Calculate maximum side growth from the actual hem/base
        # width.
        #
        # Example:
        #     base_width = 400
        #     ratio = 0.11
        #
        #     maximum growth on EACH side = 44 pixels
        # ------------------------------------------------------------
        max_side_growth_px = (
            base_width *
            max_side_growth_ratio
        )

        # Start the final Kurta mask from the cleaned shirt mask.
        kurta_mask = clean_mask.copy()

        # ------------------------------------------------------------
        # STAGE 10
        # Generate the new Kurta extension row by row.
        # ------------------------------------------------------------
        extension_height = max(
            1,
            new_bottom - y_bottom
        )

        curve_height = max(
            4,
            int(
                round(
                    extension_height *
                    bottom_curve_ratio
                )
            )
        )

        curve_start = max(
            y_bottom + 1,
            new_bottom - curve_height + 1
        )

        for y in range(
            y_bottom + 1,
            new_bottom + 1
        ):

            # --------------------------------------------------------
            # Progress through the extension:
            #
            # 0.0 = immediately below the real hem
            # 1.0 = final bottom
            # --------------------------------------------------------
            p = (
                (y - y_bottom) /
                extension_height
            )

            # --------------------------------------------------------
            # Quadratic ease-in:
            # Small growth near the shirt hem and stronger growth
            # toward the bottom.
            # --------------------------------------------------------
            eased_p = p * p

            growth = (
                max_side_growth_px *
                eased_p
            )

            # --------------------------------------------------------
            # Expand both sides from the detected base width.
            # --------------------------------------------------------
            left = int(
                round(
                    base_left - growth
                )
            )

            right = int(
                round(
                    base_right + growth
                )
            )

            # --------------------------------------------------------
            # Round only the bottom portion.
            # --------------------------------------------------------
            if y >= curve_start:

                curve_p = (
                    (y - curve_start) /
                    max(
                        1,
                        new_bottom -
                        curve_start
                    )
                )

                radius = max(
                    2,
                    int(
                        round(
                            curve_height
                        )
                    )
                )

                left, right = (
                    self._bottom_rounding(
                        left,
                        right,
                        curve_p,
                        radius
                    )
                )

            # --------------------------------------------------------
            # Keep coordinates inside the image.
            # --------------------------------------------------------
            left = max(
                0,
                left
            )

            right = min(
                img_width - 1,
                right
            )

            # --------------------------------------------------------
            # Add the new white extension row.
            # --------------------------------------------------------
            if left <= right:

                kurta_mask[
                    y,
                    left:right + 1
                ] = 1

        # ------------------------------------------------------------
        # STAGE 11
        # Restore the original mask data type.
        # ------------------------------------------------------------
        if original_dtype == bool:

            return kurta_mask.astype(
                bool
            )

        return (
            kurta_mask * 255
        ).astype(
            original_dtype
        )

    # ================================================================
    # 13. PUBLIC METHOD
    # ================================================================
    def get_garment_mask(
        self,
        shirt_mask,
        garment_type="shirt"
    ):
        """
        Return the requested garment mask.
        """

        if garment_type.lower() == "kurta":

            return self.shirt_to_kurta_mask(
                shirt_mask,
                extend_height_ratio=0.40,
                start_offset_ratio=0.05,
                max_side_growth_ratio=0.11,
                bottom_curve_ratio=0.12
            )

        return shirt_mask