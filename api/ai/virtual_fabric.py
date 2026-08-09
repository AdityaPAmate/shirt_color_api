"""
File:
    api/ai/virtual_fabric.py

Purpose:
    Creates a large virtual cloth from the uploaded fabric image using
    period-aligned seamless tiling. Never stretches the fabric.
"""

import cv2
import numpy as np
from pathlib import Path


class VirtualFabric:

    def __init__(self):
        pass

    def resize_for_tiling(self, fabric_image, scale_factor=1.0):
        if scale_factor <= 0:
            raise ValueError("scale_factor must be greater than zero.")

        if abs(scale_factor - 1.0) < 0.001:
            return fabric_image

        h, w = fabric_image.shape[:2]
        new_w = max(1, int(w * scale_factor))
        new_h = max(1, int(h * scale_factor))

        return cv2.resize(
            fabric_image,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )

    def compute_scale_factor(self, fabric_image, repeat_size):
        if repeat_size is None:
            return 1.0

        image_width = fabric_image.shape[1]

        if repeat_size > image_width * 0.40:
            return 0.50
        if repeat_size > image_width * 0.25:
            return 0.70
        if repeat_size > image_width * 0.15:
            return 0.85

        return 1.0

    def generate(
            self,
            fabric_image,
            target_width,
            target_height,
            repeat_size=None,
            repeat_size_y=None
    ):
        BASE_DIR = Path(__file__).resolve().parents[2]
        DEBUG_FOLDER = BASE_DIR / "test_images" / "debug"
        DEBUG_FOLDER.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug0_before_scale.png"),
            fabric_image
        )

        scale_factor = self.compute_scale_factor(fabric_image, repeat_size)
        fabric_image = self.resize_for_tiling(fabric_image, scale_factor)

        if repeat_size:
            repeat_size = max(1, int(repeat_size * scale_factor))
        if repeat_size_y:
            repeat_size_y = max(1, int(repeat_size_y * scale_factor))

        print(f"Scale factor applied : {scale_factor}")
        print(f"Repeat size (x) after scaling : {repeat_size}")
        print(f"Repeat size (y) after scaling : {repeat_size_y}")

        h, w = fabric_image.shape[:2]

        if repeat_size and repeat_size > 0 and repeat_size <= w:
            crop_w = (w // repeat_size) * repeat_size
        else:
            crop_w = w

        if repeat_size_y and repeat_size_y > 0 and repeat_size_y <= h:
            crop_h = (h // repeat_size_y) * repeat_size_y
        else:
            crop_h = h

        crop_w = max(crop_w, 1)
        crop_h = max(crop_h, 1)

        fabric_image = fabric_image[0:crop_h, 0:crop_w]

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug1_period_aligned_tile_unit.png"),
            fabric_image
        )

        print(f"Period-aligned tile unit size (w x h) : {crop_w} x {crop_h}")

        fabric_image = self.make_edges_seamless(fabric_image, blend_frac=0.10)

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug1b_edge_blended_tile_unit.png"),
            fabric_image
        )

        tiles_y = int(np.ceil(target_height / fabric_image.shape[0])) + 1
        tiles_x = int(np.ceil(target_width / fabric_image.shape[1])) + 1

        tiled = np.tile(fabric_image, (tiles_y, tiles_x, 1))
        tiled = tiled[0:target_height, 0:target_width]

        cv2.imwrite(
            str(DEBUG_FOLDER / "debug2_final_tiled_fabric.png"),
            tiled
        )

        return tiled

    def make_edges_seamless(self, fabric_image, blend_frac=0.10):
        img = fabric_image.astype(np.float32)
        h, w = img.shape[:2]

        bw = max(2, int(w * blend_frac))
        bh = max(2, int(h * blend_frac))

        left_strip = img[:, :bw].copy()
        right_strip = img[:, -bw:].copy()
        alpha = np.linspace(0, 1, bw).reshape(1, bw, 1)
        img[:, -bw:] = right_strip * (1 - alpha) + left_strip * alpha

        top_strip = img[:bh, :].copy()
        bottom_strip = img[-bh:, :].copy()
        beta = np.linspace(0, 1, bh).reshape(bh, 1, 1)
        img[-bh:, :] = bottom_strip * (1 - beta) + top_strip * beta

        return np.clip(img, 0, 255).astype(np.uint8)