# -*- coding: utf-8 -*-
"""Lightweight type / region helpers for Stage A (no LLM)."""
from __future__ import absolute_import, division, print_function, unicode_literals

import re

# Map free-form type text keywords -> canonical MVTec-ish tokens.
TYPE_ALIASES = {
    "scratch": ["scratch", "scratched", "abrasion", "scuff"],
    "crack": ["crack", "cracked", "fracture", "broken", "break"],
    "hole": ["hole", "puncture", "poke"],
    "cut": ["cut", "slash", "incision"],
    "contamination": ["contaminat", "dirt", "stain", "foreign", "metal contamination", "pollution"],
    "thread": ["thread", "loose thread", "fray"],
    "color": ["color", "colour", "discolor", "discolour"],
    "glue": ["glue", "adhesive"],
    "bent": ["bent", "bend", "deform", "warped"],
    "squeeze": ["squeeze", "squeezed", "compressed"],
    "print": ["imprint", "print", "misprint", "faulty imprint", "printing"],
    "missing": ["missing", "absent", "incomplete", "lack"],
    "misplaced": ["misplaced", "misalign", "offset", "skew"],
    "bristle": ["bristle", "brush"],
    "dent": ["dent", "pit", "hollow"],
    "burnt": ["burn", "scorch", "char"],
    "combined": ["combined", "multiple"],
}

REGION_KEYWORDS = {
    "top": ["top", "upper", "above"],
    "bottom": ["bottom", "lower", "below"],
    "left": ["left"],
    "right": ["right"],
    "center": ["center", "centre", "middle", "central"],
    "edge": ["edge", "border", "periphery", "rim", "corner"],
    "head": ["head", "neck", "tip"],
    "body": ["body", "handle", "shaft", "surface"],
}


def normalize_text(s):
    if not s:
        return ""
    s = s.lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonicalize_type_tokens(text):
    """Return set of canonical type tokens found in free text."""
    t = normalize_text(text)
    found = set()
    for canon, aliases in TYPE_ALIASES.items():
        for a in aliases:
            if a in t:
                found.add(canon)
                break
    # also keep raw words for soft overlap
    return found


def type_match_score(pred_type, gt_defect_folder):
    """Soft match predicted <type> vs MVTec defect folder name.

    Returns float in [0, 1], or None if not applicable (good / missing).
    """
    if not gt_defect_folder or gt_defect_folder in ("good", "ok", "normal"):
        return None
    if not pred_type:
        return 0.0

    gt = normalize_text(gt_defect_folder)
    pred = normalize_text(pred_type)

    # exact / substring
    if gt in pred or pred in gt:
        return 1.0

    gt_tokens = canonicalize_type_tokens(gt)
    # also split folder: scratch_neck -> scratch
    for part in gt.split():
        gt_tokens |= canonicalize_type_tokens(part)
    # folder may be scratch_neck without spaces after normalize -> already spaced
    for part in gt_defect_folder.lower().split("_"):
        gt_tokens |= canonicalize_type_tokens(part)

    pred_tokens = canonicalize_type_tokens(pred)
    if not gt_tokens and not pred_tokens:
        # fallback token overlap on words
        gw = set(gt.split())
        pw = set(pred.split())
        if not gw or not pw:
            return 0.0
        return len(gw & pw) / float(len(gw | pw))

    if not gt_tokens:
        return 0.0
    if not pred_tokens:
        return 0.0
    inter = gt_tokens & pred_tokens
    if inter:
        return len(inter) / float(len(gt_tokens | pred_tokens))
    return 0.0


def extract_region_tags(location_text):
    t = normalize_text(location_text)
    tags = set()
    for canon, aliases in REGION_KEYWORDS.items():
        for a in aliases:
            if re.search(r"\b" + re.escape(a) + r"\b", t):
                tags.add(canon)
                break
    return tags


def mask_region_tags(mask_arr):
    """Infer coarse region tags from a binary mask (H,W) or (H,W,C)."""
    import numpy as np

    m = np.asarray(mask_arr)
    if m.ndim == 3:
        m = m.max(axis=2)
    m = m > 0
    if not m.any():
        return set()
    ys, xs = np.where(m)
    h, w = m.shape[:2]
    cy, cx = float(ys.mean()), float(xs.mean())
    tags = set()
    if cy < h / 3.0:
        tags.add("top")
    elif cy > 2.0 * h / 3.0:
        tags.add("bottom")
    else:
        tags.add("center")
    if cx < w / 3.0:
        tags.add("left")
    elif cx > 2.0 * w / 3.0:
        tags.add("right")
    else:
        tags.add("center")
    # edge if mass near border
    edge_ratio = (
        (ys < h * 0.08).mean()
        + (ys > h * 0.92).mean()
        + (xs < w * 0.08).mean()
        + (xs > w * 0.92).mean()
    )
    if edge_ratio > 0.25:
        tags.add("edge")
    return tags


def location_mask_score(location_text, mask_arr):
    """Soft overlap between location text regions and mask regions. [0,1] or None."""
    if mask_arr is None or not location_text:
        return None
    pred = extract_region_tags(location_text)
    gt = mask_region_tags(mask_arr)
    if not pred or not gt:
        return 0.0 if pred or gt else None
    return len(pred & gt) / float(len(pred | gt))
