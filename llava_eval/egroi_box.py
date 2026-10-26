"""EG-RoI style box expansion (pure Python, no torch/llava imports).

Kept separate from ``attn_crop`` so offline crop export can run without loading
the model stack.
"""

from __future__ import annotations


def expand_box_eg_roi(box, image_wh, min_side=384):
    """Expand an ``xyxy`` box into an EG-RoI style square ROI.

    ``image_wh`` is ``(width, height)`` in ORIGINAL pixels. The returned box is a
    square whose side is ``max(min_side, box_w, box_h)`` (clamped so it always
    fits inside the image), centred on the original box and shifted back into
    frame when it would cross a border. This mirrors the training-time EG-RoI
    convention (square crop, >= 384 px before resize) so a tiny attention blob
    does not get upscaled into a blurry crop.
    """
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    img_w, img_h = float(image_wh[0]), float(image_wh[1])
    if img_w <= 1 or img_h <= 1:
        return [x0, y0, x1, y1]

    max_side = min(img_w, img_h)
    side = max(float(min_side), x1 - x0, y1 - y0)
    side = min(side, max_side)
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    half = side * 0.5
    nx0, ny0, nx1, ny1 = cx - half, cy - half, cx + half, cy + half

    if nx0 < 0.0:
        nx1 -= nx0
        nx0 = 0.0
    if ny0 < 0.0:
        ny1 -= ny0
        ny0 = 0.0
    if nx1 > img_w:
        nx0 -= nx1 - img_w
        nx1 = img_w
    if ny1 > img_h:
        ny0 -= ny1 - img_h
        ny1 = img_h
    nx0 = max(0.0, nx0)
    ny0 = max(0.0, ny0)
    nx1 = min(img_w, nx1)
    ny1 = min(img_h, ny1)
    return [nx0, ny0, nx1, ny1]
