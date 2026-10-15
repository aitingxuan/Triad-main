"""Layout + attention overlay helpers for the Triad P0 attention probe.

Pure NumPy/Pillow. Token grids are spatial anchors (not full receptive fields).
Newline tokens have no pixels and are skipped when drawing.
"""

from __future__ import annotations

import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def safe_stem(sample: dict, idx: int) -> str:
    qid = sample.get("id") or sample.get("question_id") or f"idx{idx:04d}"
    stem = re.sub(r"[^\w.\-]+", "_", str(qid))
    return stem[:120] or f"sample_{idx:04d}"


def base_grid_hw(n_base: int) -> Tuple[int, int]:
    if n_base <= 0:
        return 0, 0
    side = int(round(math.sqrt(n_base)))
    if side * side == n_base:
        return side, side
    for h in range(side, 0, -1):
        if n_base % h == 0:
            return h, n_base // h
    return 1, n_base


def infer_anyres_hw_newline(n_anyres: int) -> Optional[Tuple[int, int]]:
    """Factorize n_anyres = H * (W + 1) row-newline layout."""
    if n_anyres <= 1:
        return None
    best = None
    for h in range(1, n_anyres + 1):
        if n_anyres % h != 0:
            continue
        w = n_anyres // h - 1
        if w <= 0:
            continue
        score = abs(h - w)
        if best is None or score < best[0]:
            best = (score, h, w)
            if score == 0:
                break
    if best is None:
        return None
    return best[1], best[2]


def llava_anyres_spatial_hw(
    image_size_wh: Tuple[int, int],
    grid_pinpoints,
    vision_image_size: int,
    patches_per_side: int,
    max_num_patches: int = 9,
    n_anyres: Optional[int] = None,
) -> Optional[Tuple[int, int, str]]:
    """Mirror llava_arch unpad + optional shrink → spatial (H, W) without newlines."""
    try:
        from llava.mm_utils import get_anyres_image_grid_shape
    except Exception:
        get_anyres_image_grid_shape = None

    if get_anyres_image_grid_shape is not None and grid_pinpoints and vision_image_size and patches_per_side:
        try:
            num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                image_size_wh, grid_pinpoints, vision_image_size
            )
            unit = int(patches_per_side)
            cur_h = int(num_patch_height) * unit
            cur_w = int(num_patch_width) * unit
            original_width, original_height = int(image_size_wh[0]), int(image_size_wh[1])
            original_aspect_ratio = original_width / max(original_height, 1)
            current_aspect_ratio = cur_w / max(cur_h, 1)
            if original_aspect_ratio > current_aspect_ratio:
                scale_factor = cur_w / max(original_width, 1)
                new_height = int(original_height * scale_factor)
                padding = (cur_h - new_height) // 2
                h = max(1, cur_h - 2 * padding)
                w = cur_w
            else:
                scale_factor = cur_h / max(original_height, 1)
                new_width = int(original_width * scale_factor)
                padding = (cur_w - new_width) // 2
                h = cur_h
                w = max(1, cur_w - 2 * padding)
            if max_num_patches and max_num_patches > 0:
                times = math.sqrt(h * w / (max_num_patches * unit * unit))
                if times > 1.1:
                    h = max(1, int(h // times))
                    w = max(1, int(w // times))
            if n_anyres is not None and h * (w + 1) != int(n_anyres):
                fb = infer_anyres_hw_newline(int(n_anyres))
                if fb is not None:
                    return fb[0], fb[1], "factorize_mismatch"
            return h, w, "llava_unpad"
        except Exception:
            pass
    if n_anyres is not None:
        fb = infer_anyres_hw_newline(int(n_anyres))
        if fb is not None:
            return fb[0], fb[1], "factorize_fallback"
    return None


def build_views(
    *,
    image_size_wh: Tuple[int, int],
    n_base: int,
    n_anyres: int,
    roi_lens: Sequence[int],
    roi_spatial_shape: Optional[Sequence[int]],
    bboxes: Sequence[Sequence[int]],
    anyres_hw: Optional[Tuple[int, int]],
    patch_size: int,
    vis_start: int,
) -> List[dict]:
    """Build per-view layout metadata for BASE / ANYRES / ROI token spans."""
    width, height = map(int, image_size_wh)
    views: List[dict] = []
    offset = 0

    gh, gw = base_grid_hw(n_base)
    if n_base > 0 and gh > 0 and gw > 0:
        views.append({
            "name": "base",
            "source_box_xyxy": [0, 0, width, height],
            "processed_size": [gw * patch_size, gh * patch_size],
            "grid_shape": [gh, gw],
            "token_row_stride": gw,
            "token_offset": offset,
            "token_count": n_base,
            "sequence_span": [vis_start + offset, vis_start + offset + n_base],
        })
        offset += n_base

    if n_anyres > 0:
        if anyres_hw is None:
            anyres_hw = infer_anyres_hw_newline(n_anyres)
        if anyres_hw is None:
            raise ValueError(f"Cannot infer anyres grid for n_anyres={n_anyres}")
        ah, aw = int(anyres_hw[0]), int(anyres_hw[1])
        need = ah * (aw + 1)
        if need != n_anyres:
            fb = infer_anyres_hw_newline(n_anyres)
            if fb is None:
                raise ValueError(f"anyres token count {n_anyres} is not H*(W+1)")
            ah, aw = fb
        views.append({
            "name": "anyres",
            "source_box_xyxy": [0, 0, width, height],
            "processed_size": [aw * patch_size, ah * patch_size],
            "grid_shape": [ah, aw],
            "token_row_stride": aw + 1,
            "token_offset": offset,
            "token_count": n_anyres,
            "sequence_span": [vis_start + offset, vis_start + offset + n_anyres],
            "row_newline_count": ah,
        })
        offset += n_anyres

    roi_shape = list(roi_spatial_shape) if roi_spatial_shape is not None else None
    for i, n_roi in enumerate(roi_lens):
        n_roi = int(n_roi)
        if n_roi <= 0:
            continue
        if roi_shape is not None and len(roi_shape) == 3 and i < int(roi_shape[0]):
            rh, rw = int(roi_shape[1]), int(roi_shape[2])
        else:
            rh, rw = base_grid_hw(n_roi)
        if rh * rw != n_roi:
            rh, rw = base_grid_hw(n_roi)
        if i < len(bboxes) and bboxes[i] is not None and len(bboxes[i]) >= 4:
            x0, y0, x1, y1 = [int(v) for v in bboxes[i][:4]]
            # Legacy Triad boxes are often inclusive; make PIL crop half-open.
            if x1 >= x0 and y1 >= y0:
                x1 = min(width, x1 + 1)
                y1 = min(height, y1 + 1)
            box = [max(0, x0), max(0, y0), x1, y1]
        else:
            box = [0, 0, width, height]
        views.append({
            "name": f"roi_{i}",
            "source_box_xyxy": box,
            "processed_size": [rw * patch_size, rh * patch_size],
            "grid_shape": [rh, rw],
            "token_row_stride": rw,
            "token_offset": offset,
            "token_count": n_roi,
            "sequence_span": [vis_start + offset, vis_start + offset + n_roi],
        })
        offset += n_roi

    return views


def _spatial_indices(view: dict) -> np.ndarray:
    rows, cols = view["grid_shape"]
    stride = int(view.get("token_row_stride", cols))
    local = (
        np.arange(rows)[:, None] * stride + np.arange(cols)[None, :]
    ).reshape(-1)
    return np.asarray(view["token_offset"], dtype=np.int64) + local


def view_raw_spatial_grid(view: dict, visual_scores: np.ndarray) -> np.ndarray:
    """Return (H, W) non-negative spatial attention scores (not normalized)."""
    scores = np.asarray(visual_scores, dtype=np.float32).reshape(-1)
    idx = _spatial_indices(view)
    if idx.size and int(idx.max()) >= scores.size:
        raise ValueError(
            f"Layout needs index {int(idx.max())}, but scores length is {scores.size}"
        )
    rows, cols = int(view["grid_shape"][0]), int(view["grid_shape"][1])
    raw = np.maximum(scores[idx], 0).astype(np.float32)
    return raw.reshape(rows, cols)


def _box_to_grid_mask(
    box_xyxy: Sequence[float],
    view_box_xyxy: Sequence[float],
    grid_hw: Tuple[int, int],
) -> np.ndarray:
    """Boolean (H,W) mask: spatial cells whose centers fall inside box (image xyxy)."""
    h, w = int(grid_hw[0]), int(grid_hw[1])
    vx0, vy0, vx1, vy1 = [float(v) for v in view_box_xyxy[:4]]
    bw = max(vx1 - vx0, 1e-6)
    bh = max(vy1 - vy0, 1e-6)
    bx0, by0, bx1, by1 = [float(v) for v in box_xyxy[:4]]
    # cell centers in image coords
    ys = vy0 + (np.arange(h) + 0.5) / h * bh
    xs = vx0 + (np.arange(w) + 0.5) / w * bw
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return (xx >= bx0) & (xx < bx1) & (yy >= by0) & (yy < by1)


def summarize_spatial_concentration(
    view: dict,
    visual_scores: np.ndarray,
    *,
    roi_boxes_xyxy: Optional[Sequence[Sequence[float]]] = None,
    defect_mask: Optional[np.ndarray] = None,
    border_frac: float = 0.10,
    corner_frac: float = 0.15,
    center_radius_frac: float = 0.50,
    topk_fracs: Sequence[float] = (0.01, 0.05, 0.10),
) -> Dict[str, float]:
    """Spatial concentration / location stats for one view (view-normalized).

    Metrics are computed on the view's spatial tokens only (newlines skipped),
    after renormalizing those scores to a probability map P.
    """
    grid = view_raw_spatial_grid(view, visual_scores)
    h, w = grid.shape
    n = int(h * w)
    flat = grid.reshape(-1).astype(np.float64)
    total = float(flat.sum())
    out: Dict[str, float] = {
        "n_spatial": float(n),
        "raw_sum": round(total, 8),
    }
    if n <= 0 or total <= 0 or not np.isfinite(total):
        out.update({
            "entropy_norm": 0.0,
            "gini": 0.0,
            "hhi": 0.0,
            "centroid_offset": 0.0,
            "border_mass": 0.0,
            "corner_mass": 0.0,
            "center_mass": 0.0,
            "roi_overlap": 0.0,
            "mask_overlap": 0.0,
        })
        for frac in topk_fracs:
            out[f"top{int(round(frac * 100))}pct"] = 0.0
        return out

    p = flat / total
    # Entropy (normalized by log N)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    out["entropy_norm"] = round(ent / math.log(n), 6) if n > 1 else 0.0
    # Gini of attention mass
    sorted_p = np.sort(p)
    idx = np.arange(1, n + 1, dtype=np.float64)
    out["gini"] = round(float((2.0 * (idx * sorted_p).sum()) / (n * sorted_p.sum()) - (n + 1) / n), 6)
    out["hhi"] = round(float((p * p).sum()), 6)
    for frac in topk_fracs:
        k = max(1, int(math.ceil(frac * n)))
        out[f"top{int(round(frac * 100))}pct"] = round(float(np.sort(p)[-k:].sum()), 6)

    # Location: centroid offset, border / corner / center mass
    yy, xx = np.meshgrid(
        (np.arange(h) + 0.5) / h,
        (np.arange(w) + 0.5) / w,
        indexing="ij",
    )
    p2 = p.reshape(h, w)
    cy = float((p2 * yy).sum())
    cx = float((p2 * xx).sum())
    out["centroid_yx"] = [round(cy, 6), round(cx, 6)]
    out["centroid_offset"] = round(math.sqrt((cy - 0.5) ** 2 + (cx - 0.5) ** 2) / math.sqrt(0.5), 6)

    bf = float(np.clip(border_frac, 0.0, 0.49))
    border = (yy < bf) | (yy > 1.0 - bf) | (xx < bf) | (xx > 1.0 - bf)
    out["border_mass"] = round(float(p2[border].sum()), 6)

    cf = float(np.clip(corner_frac, 0.0, 0.5))
    corners = (
        ((yy < cf) & (xx < cf))
        | ((yy < cf) & (xx > 1.0 - cf))
        | ((yy > 1.0 - cf) & (xx < cf))
        | ((yy > 1.0 - cf) & (xx > 1.0 - cf))
    )
    out["corner_mass"] = round(float(p2[corners].sum()), 6)

    rf = float(np.clip(center_radius_frac, 0.0, 1.0))
    center = ((yy - 0.5) ** 2 + (xx - 0.5) ** 2) <= (rf * rf)
    out["center_mass"] = round(float(p2[center].sum()), 6)

    # ROI overlap: mass whose cell center falls in any ROI box (image coords)
    view_box = view.get("source_box_xyxy") or [0, 0, 1, 1]
    if roi_boxes_xyxy:
        roi_m = np.zeros((h, w), dtype=bool)
        for box in roi_boxes_xyxy:
            if box is None or len(box) < 4:
                continue
            roi_m |= _box_to_grid_mask(box, view_box, (h, w))
        out["roi_overlap"] = round(float(p2[roi_m].sum()), 6) if roi_m.any() else 0.0
        out["roi_area_frac"] = round(float(roi_m.mean()), 6)
    else:
        out["roi_overlap"] = 1.0 if str(view.get("name", "")).startswith("roi") else 0.0
        out["roi_area_frac"] = 1.0 if str(view.get("name", "")).startswith("roi") else 0.0

    # Defect-mask overlap (TP/FN); empty/None → 0
    if defect_mask is not None:
        m = np.asarray(defect_mask)
        if m.ndim == 3:
            m = m.max(axis=-1)
        m = (m > 0).astype(np.float32)
        if m.size:
            # Resize mask to grid with area averaging via PIL
            mh, mw = m.shape[:2]
            # Map view crop into mask, then to grid
            vx0, vy0, vx1, vy1 = [int(round(v)) for v in view_box[:4]]
            vx0, vy0 = max(0, vx0), max(0, vy0)
            vx1, vy1 = min(mw, max(vx0 + 1, vx1)), min(mh, max(vy0 + 1, vy1))
            crop = m[vy0:vy1, vx0:vx1]
            if crop.size:
                small = np.asarray(
                    Image.fromarray((crop * 255).astype(np.uint8)).resize(
                        (w, h), Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32,
                ) / 255.0
                cell = small >= 0.5
                out["mask_overlap"] = round(float(p2[cell].sum()), 6)
                out["mask_area_frac"] = round(float(cell.mean()), 6)
            else:
                out["mask_overlap"] = 0.0
                out["mask_area_frac"] = 0.0
        else:
            out["mask_overlap"] = 0.0
            out["mask_area_frac"] = 0.0
    else:
        out["mask_overlap"] = 0.0
        out["mask_area_frac"] = 0.0

    return out


def summarize_spatial_views(
    views: Sequence[dict],
    visual_scores: np.ndarray,
    *,
    roi_boxes_xyxy: Optional[Sequence[Sequence[float]]] = None,
    defect_mask: Optional[np.ndarray] = None,
) -> Dict[str, dict]:
    """Per-view spatial concentration dict keyed by view name."""
    return {
        str(v["name"]): summarize_spatial_concentration(
            v,
            visual_scores,
            roi_boxes_xyxy=roi_boxes_xyxy,
            defect_mask=defect_mask,
        )
        for v in views
    }


def _jet_rgb(heat01: np.ndarray) -> np.ndarray:
    x = np.clip(heat01, 0.0, 1.0)
    return np.stack([
        np.clip(1.5 - np.abs(4 * x - 3), 0, 1),
        np.clip(1.5 - np.abs(4 * x - 2), 0, 1),
        np.clip(1.5 - np.abs(4 * x - 1), 0, 1),
    ], axis=-1) * 255


def _resize_square(img: Image.Image, display_size: int) -> Image.Image:
    """Fit inside display_size×display_size, then pad to exact square."""
    display_size = max(64, int(display_size))
    w, h = img.size
    scale = display_size / max(w, h, 1)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (display_size, display_size), "white")
    canvas.paste(resized, ((display_size - nw) // 2, (display_size - nh) // 2))
    return canvas


def _load_font(size: int = 18):
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if os.path.isfile(name):
            try:
                return ImageFont.truetype(name, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def view_heat_grid(
    view: dict,
    visual_scores: np.ndarray,
    *,
    scale: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Return (H,W) heat in [0,1] and the scale used."""
    scores = np.asarray(visual_scores, dtype=np.float32).reshape(-1)
    idx = _spatial_indices(view)
    if idx.size and int(idx.max()) >= scores.size:
        raise ValueError(
            f"Layout needs index {int(idx.max())}, but scores length is {scores.size}"
        )
    rows, cols = view["grid_shape"]
    raw = np.maximum(scores[idx], 0).astype(np.float32)
    if scale is None:
        scale = float(raw.max()) if raw.size and float(raw.max()) > 0 else 1.0
    scale = float(scale) if scale > 0 else 1.0
    grid = (raw / scale).reshape(rows, cols)
    return np.clip(grid, 0.0, 1.0), scale


def render_view_pair(
    original: Image.Image,
    view: dict,
    visual_scores: np.ndarray,
    *,
    display_size: int = 768,
    scale: Optional[float] = None,
    title: Optional[str] = None,
    mass_note: Optional[str] = None,
) -> Image.Image:
    """One view: source (left) | overlay (right), both at display_size."""
    rows, cols = view["grid_shape"]
    width, height = view["processed_size"]
    box = view["source_box_xyxy"]
    source = original.convert("RGB").crop(box)
    source = source.resize((width, height), Image.Resampling.BILINEAR)
    grid, used_scale = view_heat_grid(view, visual_scores, scale=scale)
    heat = Image.fromarray(grid.astype(np.float32), mode="F")
    heat = np.asarray(heat.resize((width, height), Image.Resampling.BILINEAR))
    heat = np.clip(heat, 0.0, 1.0)
    overlay = np.clip(
        0.55 * np.asarray(source) + 0.45 * _jet_rgb(heat), 0, 255
    ).astype(np.uint8)
    source_d = _resize_square(source, display_size)
    overlay_d = _resize_square(Image.fromarray(overlay), display_size)

    margin = 10
    label_h = 36
    title_h = 28 if title else 0
    note_h = 22 if mass_note else 0
    canvas_w = display_size * 2 + margin
    canvas_h = title_h + label_h + display_size + note_h + 4
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = _load_font(18)
    font_sm = _load_font(14)
    y = 4
    if title:
        draw.text((6, y), title, fill="black", font=font)
        y = title_h
    name = view["name"]
    draw.text((6, y + 6), f"{name}  source", fill="black", font=font_sm)
    draw.text((display_size + margin + 6, y + 6), f"{name}  attn (scale={used_scale:.2e})", fill="black", font=font_sm)
    y0 = title_h + label_h
    canvas.paste(source_d, (0, y0))
    canvas.paste(overlay_d, (display_size + margin, y0))
    if mass_note:
        draw.text((6, y0 + display_size + 4), mass_note, fill="black", font=font_sm)
    return canvas


def build_attention_overlay(
    original: Image.Image,
    views: Sequence[dict],
    visual_scores: np.ndarray,
    *,
    title: Optional[str] = None,
    display_size: int = 640,
    norm: str = "per_view",
    max_cols: int = 3,
) -> Image.Image:
    """Equal-size panel of all views (source above, overlay below each column).

    ``norm``: ``per_view`` (default) or ``global`` shared max over spatial patches.
    """
    scores = np.asarray(visual_scores, dtype=np.float32).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError("Attention scores contain non-finite values")
    if not views:
        canvas = Image.new("RGB", (256, 64), "white")
        ImageDraw.Draw(canvas).text((8, 24), "no visual views", fill="black")
        return canvas

    global_scale = None
    if norm == "global":
        spatial_idx = np.concatenate([_spatial_indices(v) for v in views])
        patch_scores = np.maximum(scores[spatial_idx], 0) if spatial_idx.size else np.zeros(0)
        global_scale = float(patch_scores.max()) if patch_scores.size and float(patch_scores.max()) > 0 else 1.0

    cells = []
    for view in views:
        width, height = view["processed_size"]
        box = view["source_box_xyxy"]
        source = original.convert("RGB").crop(box)
        source = source.resize((width, height), Image.Resampling.BILINEAR)
        grid, used_scale = view_heat_grid(view, scores, scale=global_scale)
        heat = Image.fromarray(grid.astype(np.float32), mode="F")
        heat = np.asarray(heat.resize((width, height), Image.Resampling.BILINEAR))
        overlay = np.clip(
            0.55 * np.asarray(source) + 0.45 * _jet_rgb(np.clip(heat, 0, 1)), 0, 255
        ).astype(np.uint8)
        cells.append((
            f"{view['name']}  s={used_scale:.1e}",
            _resize_square(source, display_size),
            _resize_square(Image.fromarray(overlay), display_size),
        ))

    margin = 10
    label_h = 28
    title_h = 30 if title else 0
    n = len(cells)
    ncols = max(1, min(int(max_cols), n))
    nrows = int(math.ceil(n / ncols))
    cell_w = display_size
    cell_h = display_size * 2 + margin
    canvas_w = ncols * cell_w + (ncols - 1) * margin
    canvas_h = title_h + nrows * (label_h + cell_h) + (nrows - 1) * margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = _load_font(18)
    font_sm = _load_font(14)
    if title:
        draw.text((6, 6), title, fill="black", font=font)
    for i, (name, source, overlay) in enumerate(cells):
        r, c = divmod(i, ncols)
        x = c * (cell_w + margin)
        y = title_h + r * (label_h + cell_h + margin)
        draw.text((x + 4, y + 4), name, fill="black", font=font_sm)
        y0 = y + label_h
        canvas.paste(source, (x, y0))
        canvas.paste(overlay, (x, y0 + display_size + margin))
    return canvas


def summarize_attention_mass(
    full_scores: np.ndarray,
    views: Sequence[dict],
    *,
    vis_start: int,
    n_visual: int,
    query_index: int,
) -> Dict[str, Union[float, dict, int]]:
    """Softmax mass buckets over the full scored sequence (sums ≈ 1)."""
    scores = np.asarray(full_scores, dtype=np.float32).reshape(-1)
    seq_len = int(scores.size)
    total = float(scores.sum())
    if total <= 0 or not np.isfinite(total):
        total = 1.0
    vis_end = int(vis_start) + int(n_visual)
    q = int(query_index)

    def _sum(a: int, b: int) -> float:
        a = max(0, min(seq_len, int(a)))
        b = max(0, min(seq_len, int(b)))
        if b <= a:
            return 0.0
        return float(scores[a:b].sum() / total)

    buckets: Dict[str, float] = {
        "text_before_visual": _sum(0, vis_start),
    }
    covered = 0
    for view in views:
        off = int(view["token_offset"])
        cnt = int(view["token_count"])
        # spatial indices are relative to the packed visual block
        idx = _spatial_indices(view)
        abs_idx = (int(vis_start) + idx).astype(np.int64)
        abs_idx = abs_idx[(abs_idx >= 0) & (abs_idx < seq_len)]
        spatial = float(scores[abs_idx].sum() / total) if abs_idx.size else 0.0
        all_mass = _sum(vis_start + off, vis_start + off + cnt)
        buckets[view["name"]] = all_mass
        buckets[f"{view['name']}_spatial"] = spatial
        if view["name"] == "anyres":
            buckets["anyres_newline"] = max(0.0, all_mass - spatial)
        covered = max(covered, off + cnt)
    trailing = max(0, int(n_visual) - covered)
    if trailing:
        buckets["visual_trailing"] = _sum(vis_start + covered, vis_end)
    buckets["visual_all"] = _sum(vis_start, vis_end)

    # Prompt text after visual, then generation relative to query.
    if q < vis_end:
        # Query inside / before visual end (unusual for decision); lump rest.
        buckets["text_after_visual"] = _sum(vis_end, seq_len)
        buckets["query_self"] = float(scores[q] / total) if 0 <= q < seq_len else 0.0
        buckets["gen_before_query"] = 0.0
        buckets["after_query"] = _sum(q + 1, seq_len)
    else:
        buckets["text_after_visual"] = _sum(vis_end, q)  # includes prompt tail + gen up to query-1
        # Split prompt-tail vs gen if possible: prompt_len unknown; use vis_end.. as text_after
        # Fine-grain: mass at query and after.
        buckets["query_self"] = float(scores[q] / total) if 0 <= q < seq_len else 0.0
        buckets["after_query"] = _sum(q + 1, seq_len)
        # gen_before_query ≈ text_after_visual when query is first gen answer token
        # and prompt has little post-visual text — still report both names clearly.
        buckets["pre_query_after_visual"] = buckets["text_after_visual"]

    # Peak locations for debugging sinks.
    peak_i = int(np.argmax(scores)) if seq_len else -1
    peak_bucket = "other"
    if peak_i < vis_start:
        peak_bucket = "text_before_visual"
    elif peak_i < vis_end:
        rel = peak_i - vis_start
        peak_bucket = "visual"
        for view in views:
            a = int(view["token_offset"])
            b = a + int(view["token_count"])
            if a <= rel < b:
                peak_bucket = view["name"]
                break
    elif peak_i == q:
        peak_bucket = "query_self"
    elif peak_i > q:
        peak_bucket = "after_query"
    else:
        peak_bucket = "pre_query_after_visual"

    token0_mass = float(scores[0] / total) if seq_len else 0.0
    # De-sink: zero out token-0 (typical attention sink) and renormalize buckets.
    desink_buckets: Dict[str, float] = {}
    if seq_len > 1:
        ds = scores.copy()
        ds[0] = 0.0
        ds_total = float(ds.sum())
        if ds_total <= 0 or not np.isfinite(ds_total):
            ds_total = 1.0

        def _sum_ds(a: int, b: int) -> float:
            a = max(0, min(seq_len, int(a)))
            b = max(0, min(seq_len, int(b)))
            if b <= a:
                return 0.0
            return float(ds[a:b].sum() / ds_total)

        desink_buckets["text_before_visual"] = _sum_ds(0, vis_start)
        for view in views:
            off = int(view["token_offset"])
            cnt = int(view["token_count"])
            desink_buckets[view["name"]] = _sum_ds(vis_start + off, vis_start + off + cnt)
        desink_buckets["visual_all"] = _sum_ds(vis_start, vis_end)
        if q >= vis_end:
            desink_buckets["text_after_visual"] = _sum_ds(vis_end, q)
            desink_buckets["query_self"] = float(ds[q] / ds_total) if 0 <= q < seq_len else 0.0
        else:
            desink_buckets["text_after_visual"] = _sum_ds(vis_end, seq_len)
            desink_buckets["query_self"] = float(ds[q] / ds_total) if 0 <= q < seq_len else 0.0

    return {
        "seq_len": seq_len,
        "query_index": q,
        "vis_span": [int(vis_start), int(vis_end)],
        "sum": float(scores.sum()),
        "buckets": {k: round(float(v), 6) for k, v in buckets.items()},
        "token0_mass": round(token0_mass, 6),
        "buckets_desink": {k: round(float(v), 6) for k, v in desink_buckets.items()},
        "peak_index": peak_i,
        "peak_score": float(scores[peak_i]) if seq_len else 0.0,
        "peak_bucket": peak_bucket,
    }


def save_overlay(
    path: str,
    original: Image.Image,
    views: Sequence[dict],
    visual_scores: np.ndarray,
    *,
    title: Optional[str] = None,
    display_size: int = 640,
    norm: str = "per_view",
    max_cols: int = 3,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    img = build_attention_overlay(
        original, views, visual_scores,
        title=title, display_size=display_size, norm=norm, max_cols=max_cols,
    )
    img.save(path)
    return path


def save_layer_maps(
    sample_dir: str,
    layer_tag: str,
    original: Image.Image,
    views: Sequence[dict],
    visual_scores: np.ndarray,
    *,
    title: Optional[str] = None,
    display_size: int = 768,
    norm: str = "per_view",
    mass: Optional[dict] = None,
) -> Dict[str, str]:
    """Save per-view large pairs + an equal-size overview panel.

    Files:
      {layer_tag}_panel.png
      {layer_tag}_base.png
      {layer_tag}_anyres.png
      {layer_tag}_roi_0.png ...
    """
    os.makedirs(sample_dir, exist_ok=True)
    scores = np.asarray(visual_scores, dtype=np.float32).reshape(-1)
    paths: Dict[str, str] = {}
    buckets = (mass or {}).get("buckets") or {}

    global_scale = None
    if norm == "global":
        spatial_idx = np.concatenate([_spatial_indices(v) for v in views]) if views else np.zeros(0, dtype=np.int64)
        patch = np.maximum(scores[spatial_idx], 0) if spatial_idx.size else np.zeros(0)
        global_scale = float(patch.max()) if patch.size and float(patch.max()) > 0 else 1.0

    for view in views:
        name = view["name"]
        m = buckets.get(name)
        ms = buckets.get(f"{name}_spatial")
        note = None
        if m is not None:
            note = f"mass={m:.4f}"
            if ms is not None and name == "anyres":
                note += f"  spatial={ms:.4f}  newline={buckets.get('anyres_newline', 0):.4f}"
            elif ms is not None:
                note += f"  spatial={ms:.4f}"
        path = os.path.join(sample_dir, f"{layer_tag}_{name}.png")
        img = render_view_pair(
            original, view, scores,
            display_size=display_size,
            scale=global_scale,
            title=f"{title or layer_tag} | {name} | norm={norm}",
            mass_note=note,
        )
        img.save(path)
        paths[name] = path

    panel_path = os.path.join(sample_dir, f"{layer_tag}_panel.png")
    mass_line = ""
    if mass and "buckets" in mass:
        b = mass["buckets"]
        parts = [
            f"text_pre={b.get('text_before_visual', 0):.3f}",
            f"vis={b.get('visual_all', 0):.3f}",
            f"text/gen_pre_q={b.get('pre_query_after_visual', b.get('text_after_visual', 0)):.3f}",
            f"q_self={b.get('query_self', 0):.3f}",
            f"peak={mass.get('peak_bucket')}",
        ]
        mass_line = " | ".join(parts)
    panel_title = title or layer_tag
    if mass_line:
        panel_title = f"{panel_title}  ||  {mass_line}"
    save_overlay(
        panel_path, original, views, scores,
        title=panel_title,
        display_size=min(display_size, 640),
        norm=norm,
        max_cols=3,
    )
    paths["panel"] = panel_path
    return paths
