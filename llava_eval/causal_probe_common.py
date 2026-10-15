#!/usr/bin/env python3
"""Shared helpers for standalone causal probes (A1/A2/A3). Does not modify existing eval code."""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# Allow `python llava_eval/causal_*.py` from repo root
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
_LLAVA = os.path.join(_REPO, "LLaVA-NeXT")
if _LLAVA not in sys.path:
    sys.path.insert(0, _LLAVA)

from evaluate import data_read, read_mask  # noqa: E402
from evaluation_config import SUPPORTED_QSET  # noqa: E402
from simple_model_worker import SimpleModelWrapper  # noqa: E402

ANS_RE = re.compile(r"<answer>\s*(Yes|No)\s*</answer>", re.I | re.S)
LOC_RE = re.compile(r"<location>(.*?)</location>", re.I | re.S)
TYPE_RE = re.compile(r"<type>(.*?)</type>", re.I | re.S)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.I | re.S)


def parse_output(text: str) -> Dict[str, Any]:
    text = text or ""
    ans_m = ANS_RE.search(text)
    loc_m = LOC_RE.search(text)
    type_m = TYPE_RE.search(text)
    think_m = THINK_RE.search(text)
    answer = None
    if ans_m:
        answer = 1 if ans_m.group(1).lower() == "yes" else 0
    else:
        # v0-style fallback
        t = text.strip().lower()
        if t.startswith("a") or t == "a." or "answer: a" in t[:40]:
            answer = 1
        elif t.startswith("b") or t == "b." or "answer: b" in t[:40]:
            answer = 0
    return {
        "answer": answer,
        "location": loc_m.group(1).strip() if loc_m else None,
        "type": type_m.group(1).strip() if type_m else None,
        "think": think_m.group(1).strip() if think_m else None,
        "raw": text,
    }


def resolve_prompt(qs_tmp, subset: str) -> str:
    if qs_tmp is None:
        raise ValueError("prompt template is None")
    if isinstance(qs_tmp, dict):
        if subset in qs_tmp:
            return qs_tmp[subset]
        if "default" in qs_tmp:
            return qs_tmp["default"]
        raise KeyError(f"subset {subset} not in prompt config")
    return qs_tmp


def load_prompt_config(ds: str, text_src: str):
    if not text_src.startswith("config:"):
        raise ValueError("causal probes currently only support --text_src config:VERSION")
    v = text_src.replace("config:", "", 1)
    assert ds in SUPPORTED_QSET, f"{ds} not in eval config"
    assert v in SUPPORTED_QSET[ds], f"{ds} has no prompt version {v}"
    return SUPPORTED_QSET[ds][v], v


def infer_subset(sample: dict) -> str:
    if sample.get("subset"):
        return str(sample["subset"])
    origin = sample.get("origin_path") or sample.get("image") or ""
    return str(origin).split("/")[0]


def resolve_gt_mask_path(origin_path: str, mvtec_root: str) -> Optional[str]:
    """Map MVTec origin_path like 'screw/test/scratch_neck/006.png' -> GT mask file."""
    if not origin_path:
        return None
    parts = origin_path.replace("\\", "/").split("/")
    if len(parts) < 4:
        return None
    cls, _split, defect, fn = parts[0], parts[1], parts[2], parts[3]
    if defect in ("good", "ok", "normal"):
        return None
    base, _ = os.path.splitext(fn)
    cands = [
        os.path.join(mvtec_root, cls, "ground_truth", defect, f"{base}_mask.png"),
        os.path.join(mvtec_root, cls, "ground_truth", defect, fn),
        os.path.join(mvtec_root, cls, "ground_truth", defect, f"{base}.png"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return None


def load_binary_mask(path: str, size_wh: Tuple[int, int]) -> np.ndarray:
    """Return HxW uint8 mask in {0,1}, resized to image size (w,h)."""
    w, h = size_wh
    arr = np.array(Image.open(path).convert("L"))
    if arr.shape[0] != h or arr.shape[1] != w:
        arr = np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))
    return (arr > 0).astype(np.uint8)


def mask_bbox(mask01: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def square_expand_bbox(bbox: List[int], img_w: int, img_h: int, side: Optional[int] = None) -> List[int]:
    """Expand to a square crop box in [xmin,ymin,xmax,ymax] inclusive-ish pixel coords."""
    xmin, ymin, xmax, ymax = bbox
    bw = max(xmax - xmin + 1, 1)
    bh = max(ymax - ymin + 1, 1)
    if side is None:
        side = max(bw, bh, 64)
    side = int(min(side, img_w, img_h))
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x0 = max(0, min(x0, img_w - side))
    y0 = max(0, min(y0, img_h - side))
    return [x0, y0, x0 + side - 1, y0 + side - 1]


def random_bbox_away_from_mask(
    mask01: np.ndarray,
    side: int,
    rng: random.Random,
    max_tries: int = 80,
) -> List[int]:
    h, w = mask01.shape
    side = int(max(1, min(int(side), w, h)))
    # prefer zero-overlap with dilated defect
    dil = mask01.copy()
    # cheap dilation via max-pool like expand
    ys, xs = np.where(mask01 > 0)
    if len(xs):
        pad = max(side // 4, 8)
        y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
        dil[y0:y1, x0:x1] = 1
    # If defect+pad already fills almost everything, shrink the control box.
    if dil.mean() > 0.85:
        side = max(1, min(side, max(w // 8, 16), max(h // 8, 16)))
    for _ in range(max_tries):
        x0 = rng.randint(0, max(w - side, 0))
        y0 = rng.randint(0, max(h - side, 0))
        patch = dil[y0 : y0 + side, x0 : x0 + side]
        if patch.size and patch.mean() < 0.05:
            return [x0, y0, x0 + side - 1, y0 + side - 1]
    # fallback: farthest corner from defect centroid
    if len(xs):
        cx, cy = float(xs.mean()), float(ys.mean())
    else:
        cx, cy = w / 2.0, h / 2.0
    corners = [(0, 0), (max(w - side, 0), 0), (0, max(h - side, 0)), (max(w - side, 0), max(h - side, 0))]
    x0, y0 = max(corners, key=lambda p: (p[0] + side / 2 - cx) ** 2 + (p[1] + side / 2 - cy) ** 2)
    x0 = int(max(0, min(x0, max(w - side, 0))))
    y0 = int(max(0, min(y0, max(h - side, 0))))
    return [x0, y0, x0 + side - 1, y0 + side - 1]


def _rgb_mean(arr: np.ndarray) -> np.ndarray:
    """Mean RGB as shape (3,), safe for any H×W."""
    return arr.reshape(-1, 3).mean(axis=0)


def occlude_with_mask(img: Image.Image, mask01: np.ndarray, mode: str = "mean") -> Image.Image:
    arr = np.array(img.convert("RGB")).copy()
    h, w = arr.shape[:2]
    m = np.asarray(mask01)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[0] != h or m.shape[1] != w:
        m = np.array(Image.fromarray(m.astype(np.float32)).resize((w, h), Image.NEAREST))
    m = m.astype(bool)
    if not m.any():
        return img.copy()
    if mode == "mean":
        # IMPORTANT: arr.mean(axis=0) on HxWx3 yields (W,3), not (3,).
        # When mask covers (almost) the whole image, ~m is empty → must use full-image mean.
        if (~m).any():
            fill = _rgb_mean(arr[~m])
        else:
            fill = _rgb_mean(arr)
        arr[m] = fill.astype(arr.dtype, copy=False)
    elif mode == "noise":
        rng = np.random.default_rng(0)
        arr[m] = rng.integers(0, 256, size=(int(m.sum()), 3), dtype=np.uint8)
    elif mode == "black":
        arr[m] = 0
    else:
        raise ValueError(mode)
    return Image.fromarray(arr)


def occlude_bbox_region(img: Image.Image, bbox: List[int], mode: str = "mean") -> Image.Image:
    w, h = img.size
    if not bbox or len(bbox) < 4:
        return img.copy()
    x0, y0, x1, y1 = [int(v) for v in bbox[:4]]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))
    if x1 < x0 or y1 < y0:
        return img.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0 : y1 + 1, x0 : x1 + 1] = 1
    return occlude_with_mask(img, mask, mode=mode)


def musc_to_binary(mask_arr: np.ndarray, img_wh: Tuple[int, int]) -> np.ndarray:
    """Threshold MusC anomaly map to binary, resize to image (w,h)."""
    w, h = img_wh
    m = np.array(mask_arr)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[0] != h or m.shape[1] != w:
        m = np.array(Image.fromarray(m.astype(np.float32)).resize((w, h), Image.BILINEAR))
    m = m.astype(np.float32)
    mn, mx = float(m.min()), float(m.max())
    if mx <= mn:
        return np.zeros((h, w), dtype=np.uint8)
    thr = mn + 0.9 * (mx - mn)
    return (m >= thr).astype(np.uint8)


def stratified_pick(
    datas: Sequence[dict],
    num_samples: int,
    seed: int,
    require_gt_mask: bool = False,
    mvtec_root: Optional[str] = None,
    defect_only: bool = False,
) -> List[dict]:
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for d in datas:
        gt = int(d.get("gt", 0))
        if defect_only and gt != 1:
            continue
        subset = infer_subset(d)
        if require_gt_mask:
            if gt != 1:
                continue
            gtp = resolve_gt_mask_path(d.get("origin_path", ""), mvtec_root or "")
            if not gtp:
                continue
            d = dict(d)
            d["_gt_mask_path"] = gtp
        buckets[(subset, gt)].append(d)
    keys = sorted(buckets.keys())
    for k in keys:
        rng.shuffle(buckets[k])
    picked: List[dict] = []
    while len(picked) < num_samples and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(picked) < num_samples:
                picked.append(buckets[k].pop())
    rng.shuffle(picked)
    return picked


def load_worker(model_path: str, model_base: Optional[str] = None, use_flash_attn: bool = False):
    return SimpleModelWrapper(
        model_path,
        model_base,
        None,
        "cuda",
        use_flash_attn,
        image_aspect_ratio=None,
    )


def run_once(
    worker: SimpleModelWrapper,
    prompt: str,
    image: Image.Image,
    masks: Optional[List[Any]],
    boxes_list: Optional[List[Any]],
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.7,
) -> Dict[str, Any]:
    inputs = {
        "text": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "images": [image],
        "masks": masks,
        "boxes_list": boxes_list,
    }
    _, out = worker.single_qa(inputs)
    parsed = parse_output(out)
    return parsed


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def summarize_condition_rows(rows: List[dict]) -> Dict[str, Any]:
    """Aggregate flip / acc stats vs baseline per condition name."""
    by_cond: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    if "baseline" not in by_cond and "process_correct" not in by_cond:
        base_key = next(iter(by_cond))
    else:
        base_key = "baseline" if "baseline" in by_cond else "process_correct"
    base_map = {r["id"]: r for r in by_cond[base_key]}
    out = {}
    for cond, items in sorted(by_cond.items()):
        n = len(items)
        decided = [r for r in items if r.get("answer") is not None]
        correct = [r for r in decided if r["answer"] == r["gt"]]
        flips = 0
        flip_ids = []
        for r in items:
            b = base_map.get(r["id"])
            if not b:
                continue
            if b.get("answer") is not None and r.get("answer") is not None and b["answer"] != r["answer"]:
                flips += 1
                flip_ids.append(r["id"])
        out[cond] = {
            "n": n,
            "decided": len(decided),
            "acc": (len(correct) / len(decided)) if decided else None,
            "yes_rate": (sum(r["answer"] == 1 for r in decided) / len(decided)) if decided else None,
            "answer_flip_vs_baseline": flips,
            "answer_flip_rate": flips / max(len(items), 1),
            "flip_ids": flip_ids[:20],
        }
    out["_baseline_condition"] = base_key
    return out
