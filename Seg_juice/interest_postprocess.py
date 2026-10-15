"""Map LogSAD-style juice_bottle merge labels → PGRB closed-set interests."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Raw CLIP+SAM classes in sam_clip_juice_bottle.CLASS_NAMES
RAW_GLASS, RAW_LIQUID, RAW_FRUIT, RAW_LABEL, RAW_BG = 0, 1, 2, 3, 4

# PGRB interest id map (stored in interests.png)
INTEREST_ID = {
    "background": 0,
    "juice": 1,
    "top_label": 2,
    "bottom_label": 3,
    "bottle_body": 4,
}
INTEREST_NAMES = ["background", "juice", "top_label", "bottom_label", "bottle_body"]

# Visualization (RGB)
INTEREST_COLORS = np.array(
    [
        [40, 40, 40],      # background
        [40, 40, 220],     # juice
        [40, 200, 220],    # top_label
        [40, 180, 120],    # bottom_label
        [180, 120, 40],    # bottle_body
    ],
    dtype=np.uint8,
)


def _bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _split_label_mask(label_mask: np.ndarray, min_cc_area: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a single 'label' mask into top / bottom by connected components (fallback: y-median)."""
    h, w = label_mask.shape
    top = np.zeros((h, w), dtype=bool)
    bottom = np.zeros((h, w), dtype=bool)
    if not label_mask.any():
        return top, bottom

    binary = label_mask.astype(np.uint8)
    n, cc, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    blobs: list[tuple[float, np.ndarray, int]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_cc_area:
            continue
        blobs.append((float(centroids[i][1]), cc == i, area))

    if len(blobs) >= 2:
        blobs.sort(key=lambda t: t[0])  # by cy ascending
        # Largest in upper half of blobs → top; largest in lower → bottom
        mid = len(blobs) // 2
        upper, lower = blobs[:mid], blobs[mid:]
        # Prefer largest blob in each half; if one half empty, take extremes
        if not upper:
            upper = [blobs[0]]
            lower = blobs[1:]
        if not lower:
            lower = [blobs[-1]]
            upper = blobs[:-1]
        top_blob = max(upper, key=lambda t: t[2])[1]
        bottom_blob = max(lower, key=lambda t: t[2])[1]
        top |= top_blob
        bottom |= bottom_blob
        # leftover label pixels: assign by y vs mid of top/bottom centroids
        assigned = top | bottom
        leftover = label_mask & ~assigned
        if leftover.any():
            cy_top = float(np.mean(np.where(top)[0])) if top.any() else 0.0
            cy_bot = float(np.mean(np.where(bottom)[0])) if bottom.any() else float(h)
            split_y = 0.5 * (cy_top + cy_bot)
            ys, xs = np.where(leftover)
            for y, x in zip(ys, xs):
                if y <= split_y:
                    top[y, x] = True
                else:
                    bottom[y, x] = True
        return top, bottom

    # 0–1 blob: split by median y of label pixels
    ys, xs = np.where(label_mask)
    mid_y = float(np.median(ys))
    top[ys, xs] = ys <= mid_y
    bottom[ys, xs] = ys > mid_y
    return top, bottom


def _quality(
    name: str,
    mask: np.ndarray,
    img_h: int,
    img_w: int,
    *,
    min_area_frac: float,
    max_area_frac: float,
) -> tuple[float, bool]:
    area = int(mask.sum())
    total = img_h * img_w
    frac = area / max(total, 1)
    if area == 0:
        return 0.0, False
    q = 1.0
    if frac < min_area_frac:
        q *= 0.4
    if frac > max_area_frac:
        q *= 0.5
    # touch image border heavily → suspicious for labels/juice
    ys, xs = np.where(mask)
    border_touch = (
        int(ys.min()) <= 1
        or int(ys.max()) >= img_h - 2
        or int(xs.min()) <= 1
        or int(xs.max()) >= img_w - 2
    )
    if name in {"top_label", "bottom_label", "juice"} and border_touch:
        q *= 0.7
    usable = q >= 0.45 and area >= 64
    return float(round(q, 4)), bool(usable)


def remap_merge_to_interests(
    merge_clip_res: np.ndarray,
    orig_hw: tuple[int, int],
    *,
    min_cc_area: int = 80,
    juice_min_frac: float = 0.01,
    juice_max_frac: float = 0.55,
    label_min_frac: float = 0.002,
    label_max_frac: float = 0.25,
    body_min_frac: float = 0.005,
    body_max_frac: float = 0.45,
) -> dict[str, Any]:
    """
    Resize raw merge (CLIP res) to original HxW and build interest id map.

    Returns dict with:
      interests_map (uint8 HxW), masks (dict name→bool), meta interests fields
    """
    orig_h, orig_w = orig_hw
    merge = cv2.resize(merge_clip_res.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    label_m = merge == RAW_LABEL
    juice_m = ((merge == RAW_LIQUID) | (merge == RAW_FRUIT)) & ~label_m
    body_m = (merge == RAW_GLASS) & ~label_m & ~juice_m
    top_m, bottom_m = _split_label_mask(label_m, min_cc_area=min_cc_area)

    # Priority: labels > juice > body > bg
    interests_map = np.zeros((orig_h, orig_w), dtype=np.uint8)
    interests_map[body_m] = INTEREST_ID["bottle_body"]
    interests_map[juice_m] = INTEREST_ID["juice"]
    interests_map[top_m] = INTEREST_ID["top_label"]
    interests_map[bottom_m] = INTEREST_ID["bottom_label"]

    masks = {
        "juice": juice_m,
        "top_label": top_m,
        "bottom_label": bottom_m,
        "bottle_body": body_m,
    }
    frac_cfg = {
        "juice": (juice_min_frac, juice_max_frac),
        "top_label": (label_min_frac, label_max_frac),
        "bottom_label": (label_min_frac, label_max_frac),
        "bottle_body": (body_min_frac, body_max_frac),
    }

    interests_meta: dict[str, Any] = {}
    for name, mask in masks.items():
        mn, mx = frac_cfg[name]
        q, usable = _quality(name, mask, orig_h, orig_w, min_area_frac=mn, max_area_frac=mx)
        bbox = _bbox_xyxy(mask)
        interests_meta[name] = {
            "id": INTEREST_ID[name],
            "bbox": bbox,
            "area": int(mask.sum()),
            "quality": q,
            "usable": usable and bbox is not None,
        }

    return {
        "interests_map": interests_map,
        "masks": masks,
        "interests": interests_meta,
        "size": [orig_h, orig_w],
    }


def colorize_interests(interests_map: np.ndarray) -> np.ndarray:
    h, w = interests_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(INTEREST_COLORS):
        rgb[interests_map == i] = color
    return rgb


def save_interest_id_png(path: Path | str, interests_map: np.ndarray) -> None:
    """Save uint8 id map as palette PNG: indices intact, but viewable in image apps."""
    img = Image.fromarray(interests_map.astype(np.uint8), mode="P")
    palette = [0] * (256 * 3)
    for i, color in enumerate(INTEREST_COLORS):
        palette[i * 3 : i * 3 + 3] = [int(c) for c in color]
    img.putpalette(palette)
    img.save(path)
