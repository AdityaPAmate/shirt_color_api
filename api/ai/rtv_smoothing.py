"""
File:
    api/ai/rtv_smoothing.py

Purpose:
    Relative Total Variation (RTV) based structure extraction.
    Separates garment SHADING/FOLD structure from FABRIC WEAVE/PRINT texture.

Reference:
    Xu, Yan, Xia, Jia - "Structure Extraction from Texture via
    Relative Total Variation", SIGGRAPH Asia 2012.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import cv2


def _compute_texture_weights(fin, sigma, eps=1e-3):
    fx = np.gradient(fin, axis=1)
    fy = np.gradient(fin, axis=0)

    ksize = int(max(3, round(sigma * 2 + 1)))
    if ksize % 2 == 0:
        ksize += 1

    wto_x = cv2.boxFilter(np.abs(fx), -1, (ksize, ksize))
    wto_y = cv2.boxFilter(np.abs(fy), -1, (ksize, ksize))

    blurred = cv2.GaussianBlur(fin, (0, 0), sigma)
    fbx = np.gradient(blurred, axis=1)
    fby = np.gradient(blurred, axis=0)
    wtl_x = np.abs(cv2.boxFilter(fbx, -1, (ksize, ksize)))
    wtl_y = np.abs(cv2.boxFilter(fby, -1, (ksize, ksize)))

    retx = (wtl_x + eps) / (wto_x + eps)
    rety = (wtl_y + eps) / (wto_y + eps)

    return retx, rety


def rtv_smooth(image_gray_float, lam=0.015, sigma=3.0, iterations=4, eps=1e-3):
    """
    image_gray_float : float64, single channel, range [0,1]
    lam   : texture removal strength
    sigma : texture scale (roughly = weave pitch in pixels)
    Returns: structure layer (folds/shading only, texture removed), range [0,1]
    """
    S = image_gray_float.copy()
    I = image_gray_float
    H, W = I.shape
    N = H * W

    for _ in range(iterations):
        wx, wy = _compute_texture_weights(S, sigma, eps)
        wx = wx.flatten()
        wy = wy.flatten()

        dx = -lam * wx
        dy = -lam * wy

        idx = np.arange(N).reshape(H, W)
        row, col, data = [], [], []
        diag = np.ones(N)

        valid_x = idx[:, :-1].flatten()
        valid_x2 = idx[:, 1:].flatten()
        wx_valid = dx.reshape(H, W)[:, :-1].flatten()
        row.extend(valid_x); col.extend(valid_x2); data.extend(wx_valid)
        row.extend(valid_x2); col.extend(valid_x); data.extend(wx_valid)
        diag[valid_x] -= wx_valid
        diag[valid_x2] -= wx_valid

        valid_y = idx[:-1, :].flatten()
        valid_y2 = idx[1:, :].flatten()
        wy_valid = dy.reshape(H, W)[:-1, :].flatten()
        row.extend(valid_y); col.extend(valid_y2); data.extend(wy_valid)
        row.extend(valid_y2); col.extend(valid_y); data.extend(wy_valid)
        diag[valid_y] -= wy_valid
        diag[valid_y2] -= wy_valid

        row.extend(np.arange(N))
        col.extend(np.arange(N))
        data.extend(diag)

        A = sp.csr_matrix((data, (row, col)), shape=(N, N))
        b = I.flatten()

        S = spsolve(A.tocsc(), b).reshape(H, W)

    return np.clip(S, 0, 1)


def extract_rtv_structure(
        person_image_bgr,
        shirt_mask,
        sigma=3.0,
        lam=0.015,
        iterations=4,
        max_dim=640
):
    mask_bin = (shirt_mask > 0).astype(np.uint8)
    ys, xs = np.where(mask_bin > 0)

    if len(ys) == 0:
        return np.ones(person_image_bgr.shape[:2], dtype=np.float32)

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    crop = person_image_bgr[y0:y1, x0:x1]
    crop_mask = mask_bin[y0:y1, x0:x1]

    # ------------------------------------------------------------
    # NEW: शर्ट नसलेला भाग (केस, मान, background) inpaint करा
    # जेणेकरून RTV ला फक्त शर्टचाच content दिसेल
    # ------------------------------------------------------------
    non_shirt_area = (crop_mask == 0).astype(np.uint8) * 255
    crop = cv2.inpaint(crop, non_shirt_area, 9, cv2.INPAINT_TELEA)

    h, w = crop.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    small = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))))
    small_mask = cv2.resize(
        crop_mask, (small.shape[1], small.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float64)
    L = lab[:, :, 0] / 255.0

    sigma_scaled = max(1.2, sigma * scale)

    structure_small = rtv_smooth(
        L, lam=lam, sigma=sigma_scaled, iterations=iterations
    )

    structure_crop = cv2.resize(
        structure_small.astype(np.float32),
        (crop.shape[1], crop.shape[0])
    )

    local_mean = cv2.GaussianBlur(structure_crop, (0, 0), 21)
    local_mean = np.clip(local_mean, 1e-3, None)
    shading_crop = structure_crop / local_mean

    # ------------------------------------------------------------
    # NEW: सुरक्षा -- शर्ट नसलेल्या pixel वर shading = 1.0 (neutral) करा
    # ------------------------------------------------------------
    shading_crop[crop_mask == 0] = 1.0

    shading_full = np.ones(person_image_bgr.shape[:2], dtype=np.float32)
    shading_full[y0:y1, x0:x1] = shading_crop

    return shading_full