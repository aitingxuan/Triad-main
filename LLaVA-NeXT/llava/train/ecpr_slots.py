"""ECPR slot composition: EG-RoI first, conditional part-ROI fill/replace.

No SAM at runtime — part boxes come from offline interest-mask cache
(``Seg_juice/juice_interest_masks_cotm``). Week-2 aux supervision fields are
attached by ``scripts/build_ecpr_sidecar.py`` and consumed by ``ecpr_aux_loss``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

Box = List[float]  # xyxy


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = map(float, a[:4])
    bx0, by0, bx1, by1 = map(float, b[:4])
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def expand_box_square(box: Sequence[float], image_wh: Optional[Tuple[int, int]] = None) -> Box:
    """Make a square crop around ``box``, clipped to image if size given."""
    x0, y0, x1, y1 = map(float, box[:4])
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    side = max(x1 - x0, y1 - y0, 1.0)
    half = side * 0.5
    nx0, ny0, nx1, ny1 = cx - half, cy - half, cx + half, cy + half
    if image_wh is not None:
        w, h = float(image_wh[0]), float(image_wh[1])
        # shift back into frame if possible
        if nx1 - nx0 > w:
            nx0, nx1 = 0.0, w
        else:
            if nx0 < 0:
                nx1 -= nx0
                nx0 = 0.0
            if nx1 > w:
                nx0 -= nx1 - w
                nx1 = w
        if ny1 - ny0 > h:
            ny0, ny1 = 0.0, h
        else:
            if ny0 < 0:
                ny1 -= ny0
                ny0 = 0.0
            if ny1 > h:
                ny0 -= ny1 - h
                ny1 = h
        nx0, ny0 = max(0.0, nx0), max(0.0, ny0)
        nx1, ny1 = min(w, nx1), min(h, ny1)
    return [nx0, ny0, nx1, ny1]


def normalize_eg_boxes(raw: Any) -> List[Box]:
    """Flatten CoTM / eval bbox payloads into a list of xyxy boxes."""
    if raw is None:
        return []
    boxes: List[Box] = []

    def _push(b: Any) -> None:
        if b is None:
            return
        if isinstance(b, (list, tuple)) and len(b) >= 4 and all(isinstance(x, (int, float)) for x in b[:4]):
            boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
            return
        if isinstance(b, (list, tuple)):
            for item in b:
                _push(item)

    _push(raw)
    return boxes


def eg_covers_part(eg_boxes: Sequence[Box], part_box: Sequence[float], tau: float) -> bool:
    """True if any EG box IoU with part >= tau (EG already covers the part)."""
    if not eg_boxes or part_box is None:
        return False
    for eg in eg_boxes:
        if box_iou(eg, part_box) >= tau:
            return True
    return False


def load_trigger_rules(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def load_mask_index(index_path: str) -> Dict[str, Dict[str, Any]]:
    """Map relative_id and several path aliases → index row."""
    by_key: Dict[str, Dict[str, Any]] = {}
    with open(index_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row.get("relative_id")
            if rid:
                by_key[rid] = row
                by_key[rid.replace("\\", "/")] = row
            ip = row.get("image_path") or ""
            if ip:
                by_key[ip] = row
                by_key[os.path.abspath(ip)] = row
                # suffix from *_juice/...
                parts = Path(ip).parts
                for i, part in enumerate(parts):
                    if part.endswith("_juice"):
                        alias = "/".join(parts[i:]).replace(".png", "").replace(".jpg", "")
                        by_key[alias] = row
                        break
    return by_key


def image_to_relative_id(image_path: str) -> Optional[str]:
    parts = Path(str(image_path).replace("\\", "/")).parts
    for i, part in enumerate(parts):
        if part.endswith("_juice"):
            return "/".join(parts[i:]).rsplit(".", 1)[0]
    return None


def lookup_mask_row(
    mask_index: Dict[str, Dict[str, Any]],
    image_path: Optional[str] = None,
    relative_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if relative_id and relative_id in mask_index:
        return mask_index[relative_id]
    if image_path:
        if image_path in mask_index:
            return mask_index[image_path]
        abs_p = os.path.abspath(image_path)
        if abs_p in mask_index:
            return mask_index[abs_p]
        rid = image_to_relative_id(image_path)
        if rid and rid in mask_index:
            return mask_index[rid]
    return None


def _image_wh_from_mask_row(mask_row: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    size = mask_row.get("size")
    if not (isinstance(size, (list, tuple)) and len(size) >= 2):
        return None
    # index stores size as [H, W] for portrait bottles (e.g. [1600, 800])
    h, w = int(size[0]), int(size[1])
    if h > w:
        return (w, h)
    return (int(size[0]), int(size[1]))


def product_from_relative_id(relative_id: Optional[str]) -> Optional[str]:
    if not relative_id:
        return None
    head = relative_id.replace("\\", "/").split("/", 1)[0]
    return head if head.endswith("_juice") else None


def build_part_priors_from_good(
    mask_index: Dict[str, Dict[str, Any]],
    products: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Box]]:
    """Median bbox per (product, interest) from usable *good* mask rows.

    Used when missing_* defects have no SAM instance (usable=false) but still
    need an expected part ROI for ECPR inject.
    """
    from statistics import median

    # Deduplicate by relative_id (index stores many path aliases → same row).
    seen = set()
    acc: Dict[str, Dict[str, List[Box]]] = {}
    for key, row in mask_index.items():
        rid = row.get("relative_id") or ""
        if rid in seen:
            continue
        if "/image/good/" not in rid.replace("\\", "/"):
            continue
        seen.add(rid)
        product = product_from_relative_id(rid)
        if not product:
            continue
        if products is not None and product not in products:
            continue
        bucket = acc.setdefault(product, {})
        for name, meta in (row.get("interests") or {}).items():
            if not meta.get("usable", False):
                continue
            bbox = meta.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            bucket.setdefault(name, []).append([float(x) for x in bbox[:4]])

    priors: Dict[str, Dict[str, Box]] = {}
    for product, by_name in acc.items():
        priors[product] = {}
        for name, boxes in by_name.items():
            if not boxes:
                continue
            priors[product][name] = [
                float(median(b[i] for b in boxes)) for i in range(4)
            ]
    return priors


def candidate_parts_for_defect(
    defect_type: Optional[str],
    rules: Dict[str, Any],
    mask_row: Optional[Dict[str, Any]],
    eg_boxes: Sequence[Box],
    tau: Optional[float] = None,
    part_priors: Optional[Dict[str, Dict[str, Box]]] = None,
) -> List[Dict[str, Any]]:
    """Return part dicts {name, box, raw_box, source} that should be injected."""
    if not defect_type or defect_type == "good" or mask_row is None:
        return []
    mapping = rules.get("defect_to_interests") or {}
    names = list(mapping.get(defect_type) or [])
    if not names:
        return []
    tau_v = float(rules.get("iou_tau", 0.25) if tau is None else tau)
    interests = mask_row.get("interests") or {}
    image_wh = _image_wh_from_mask_row(mask_row)
    allow_prior = bool(rules.get("missing_use_part_prior", True)) and (
        "missing" in str(defect_type)
    )
    product = product_from_relative_id(mask_row.get("relative_id"))
    prior_map = (part_priors or {}).get(product or "", {}) if allow_prior else {}

    out: List[Dict[str, Any]] = []
    for name in names:
        meta = interests.get(name) or {}
        bbox = None
        src = "mask"
        if meta.get("usable", False) and meta.get("bbox") and len(meta["bbox"]) >= 4:
            bbox = list(map(float, meta["bbox"][:4]))
        elif allow_prior and name in prior_map:
            bbox = list(map(float, prior_map[name][:4]))
            src = "prior"
        if bbox is None:
            continue
        if eg_covers_part(eg_boxes, bbox, tau_v):
            continue
        part_box = expand_box_square(bbox, image_wh)
        out.append(
            {
                "name": name,
                "box": part_box,
                "raw_box": bbox,
                "source": src,
            }
        )
    return out


def compose_slots(
    eg_boxes: Sequence[Box],
    part_candidates: Sequence[Dict[str, Any]],
    max_box_num: int = 4,
) -> Dict[str, Any]:
    """EG first; append parts; if full, replace lowest-IoU EG vs that part."""
    max_n = int(max_box_num)
    slots: List[Box] = [list(map(float, b[:4])) for b in eg_boxes[:max_n]]
    sources: List[str] = ["eg"] * len(slots)
    names: List[str] = [f"eg{i}" for i in range(len(slots))]
    replaced: List[Dict[str, Any]] = []

    for part in part_candidates:
        pbox = list(map(float, part["box"][:4]))
        pname = str(part.get("name", "part"))
        if len(slots) < max_n:
            slots.append(pbox)
            sources.append("part")
            names.append(pname)
            continue
        eg_idxs = [i for i, s in enumerate(sources) if s == "eg"]
        if not eg_idxs:
            break
        worst = min(eg_idxs, key=lambda i: box_iou(slots[i], pbox))
        replaced.append(
            {
                "slot": worst,
                "old_box": slots[worst],
                "old_name": names[worst],
                "part": pname,
            }
        )
        slots[worst] = pbox
        sources[worst] = "part"
        names[worst] = pname

    return {
        "boxes": slots,
        "slot_sources": sources,
        "slot_names": names,
        "replaced": replaced,
        "n_eg": sum(1 for s in sources if s == "eg"),
        "n_part": sum(1 for s in sources if s == "part"),
    }


def compose_for_sample(
    *,
    image_path: str,
    defect_type: Optional[str],
    eg_boxes_raw: Any,
    rules: Dict[str, Any],
    mask_index: Dict[str, Dict[str, Any]],
    tau: Optional[float] = None,
    max_box_num: Optional[int] = None,
    part_priors: Optional[Dict[str, Dict[str, Box]]] = None,
) -> Dict[str, Any]:
    eg_boxes = normalize_eg_boxes(eg_boxes_raw)
    max_n = int(max_box_num if max_box_num is not None else rules.get("max_box_num", 4))
    tau_v = float(rules.get("iou_tau", 0.25) if tau is None else tau)
    rid = image_to_relative_id(image_path)
    mask_row = lookup_mask_row(mask_index, image_path=image_path, relative_id=rid)
    parts = candidate_parts_for_defect(
        defect_type,
        rules,
        mask_row,
        eg_boxes,
        tau=tau_v,
        part_priors=part_priors,
    )
    composed = compose_slots(eg_boxes, parts, max_box_num=max_n)
    return {
        "relative_id": rid,
        "defect_type": defect_type,
        "has_mask": mask_row is not None,
        "eg_boxes": eg_boxes,
        "triggered_parts": [p["name"] for p in parts],
        "part_sources": [p.get("source", "mask") for p in parts],
        "candidate_parts": parts,
        "iou_tau": tau_v,
        **composed,
    }


def load_sidecar_index(sidecar_path: str) -> Dict[str, Dict[str, Any]]:
    """Map relative_id / image → sidecar row (final boxes ready for train)."""
    by_key: Dict[str, Dict[str, Any]] = {}
    with open(sidecar_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in (row.get("relative_id"), row.get("image"), row.get("image_path")):
                if key:
                    by_key[str(key)] = row
            img = row.get("image") or row.get("image_path")
            if img:
                rid = image_to_relative_id(img)
                if rid:
                    by_key[rid] = row
    return by_key


def lookup_sidecar_row(
    image_path: str,
    sidecar_index: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return sidecar row for an image path, or None if not covered."""
    rid = image_to_relative_id(image_path)
    if rid and rid in sidecar_index:
        return sidecar_index[rid]
    if image_path in sidecar_index:
        return sidecar_index[image_path]
    abs_p = os.path.abspath(image_path)
    if abs_p in sidecar_index:
        return sidecar_index[abs_p]
    return None


def apply_sidecar_boxes(
    image_path: str,
    sidecar_index: Dict[str, Dict[str, Any]],
) -> Optional[List[Box]]:
    """Return final boxes from sidecar, or None if sample not covered."""
    row = lookup_sidecar_row(image_path, sidecar_index)
    if row is None:
        return None
    boxes = row.get("boxes") or row.get("final_boxes")
    if not boxes:
        return None
    return [list(map(float, b[:4])) for b in boxes]


def sidecar_row_to_ecpr_meta(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Compact meta dict passed through the dataloader for Week-2 aux losses."""
    if not row:
        return None
    boxes = row.get("boxes") or row.get("final_boxes") or []
    return {
        "ecpr_active": True,
        "relative_id": row.get("relative_id"),
        "defect_type": row.get("defect_type"),
        "sub_dataset": row.get("sub_dataset"),
        "is_anomaly": bool(row.get("is_anomaly", row.get("defect_type") not in (None, "good"))),
        "gt_interests": list(row.get("gt_interests") or []),
        "gt_step": row.get("gt_step"),
        "gt_step_name": row.get("gt_step_name"),
        "gt_type_text": row.get("gt_type_text"),
        "gt_location_tag": row.get("gt_location_tag"),
        "slot_sources": list(row.get("slot_sources") or []),
        "slot_names": list(row.get("slot_names") or []),
        "slot_positive": list(row.get("slot_positive") or []),
        "n_part": int(row.get("n_part") or sum(1 for s in (row.get("slot_sources") or []) if s == "part")),
        "n_eg": int(row.get("n_eg") or sum(1 for s in (row.get("slot_sources") or []) if s == "eg")),
        "boxes": [list(map(float, b[:4])) for b in boxes],
    }
