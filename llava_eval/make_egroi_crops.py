"""Re-crop evaluation images from saved attention boxes, EG-RoI style.

Typical use (CPU only, no model load):

    python llava_eval/make_egroi_crops.py \
        --answers  result/.../answer.json \
        --data-dir ./data/mvtec \
        --out-dir  ./result/egroi_crops \
        --min-side 384 --save-viz

It reads ``ab_decision.box`` (or ``ab_decision.real.box`` / top-level ``box``)
from an evaluate.py answer file, expands each box to a square with side
``>= --min-side`` in ORIGINAL pixels, crops the ORIGINAL image, writes the crop
PNGs and an ``egroi_crops.json`` dataset that can be fed straight to evaluate.py:

    python llava_eval/evaluate.py ... \
        --data_path <out-dir> --question_file egroi_crops.json \
        --text_src config:v0 --ab_tau 0.5

The long-CoT tokens are used ONLY to decide where to crop; the v0 run is a pure
single-token A/B binary decision on the crop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "LLaVA-NeXT"))

from PIL import Image, ImageDraw  # noqa: E402

from egroi_box import expand_box_eg_roi  # noqa: E402


def _load_records(path):
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data


def _load_ids(path):
    if not path:
        return None
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = list(data.keys()) if all(isinstance(v, (bool, int)) for v in data.values()) else data
        return {str(x) for x in data}
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _first_image(rec):
    img = rec.get("image")
    if img is None:
        img = rec.get("src")
    if isinstance(img, (list, tuple)):
        img = img[0] if img else None
    return img


def _record_id(rec, idx):
    for k in ("id", "question_id", "sample_id"):
        if rec.get(k) is not None:
            return str(rec[k])
    return str(idx)


def _extract_box(rec):
    ab = rec.get("ab_decision") or {}
    box = ab.get("box")
    if box is None:
        box = (ab.get("real") or {}).get("box")
    if box is None:
        box = rec.get("box")
    if box is None:
        box = rec.get("origin_box")
    return box


def _resolve_image(data_dir, img):
    if img is None:
        return None
    img = str(img)
    if os.path.isabs(img):
        return img
    return os.path.join(data_dir, img)


def _safe_name(text, limit=100):
    import re
    return re.sub(r"[^\w.\-]+", "_", str(text))[:limit]


def _save_viz(path, image, box_orig, box_new):
    scale = 640.0 / max(image.width, image.height, 1)
    nw, nh = max(1, int(image.width * scale)), max(1, int(image.height * scale))
    canvas = Image.new("RGB", (nw * 2 + 10, nh + 26), "white")
    draw = ImageDraw.Draw(canvas)

    left = image.resize((nw, nh), Image.Resampling.BILINEAR).convert("RGB")
    ld = ImageDraw.Draw(left)
    sx, sy = nw / image.width, nh / image.height
    if box_orig is not None:
        x0, y0, x1, y1 = box_orig
        ld.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=(255, 215, 0), width=2)
    if box_new is not None:
        x0, y0, x1, y1 = box_new
        ld.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=(255, 0, 0), width=3)
    canvas.paste(left, (0, 26))

    if box_new is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in box_new]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image.width, max(x0 + 1, x1)), min(image.height, max(y0 + 1, y1))
        crop = image.crop((x0, y0, x1, y1))
        crop.thumbnail((nw, nh), Image.Resampling.BILINEAR)
        canvas.paste(crop, (nw + 10 + (nw - crop.width) // 2, 26 + (nh - crop.height) // 2))

    draw.text((4, 6), "original + box (yellow=attn, red=EG-RoI)", fill="black")
    draw.text((nw + 14, 6), "EG-RoI crop", fill="black")
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, help="answer.json / .jsonl from evaluate.py")
    ap.add_argument("--data-dir", required=True, help="root for relative image paths")
    ap.add_argument("--out-dir", required=True, help="where crops + egroi_crops.json are written")
    ap.add_argument("--min-side", type=int, default=384, help="EG-RoI minimum square side (px)")
    ap.add_argument("--ids-file", default=None, help="optional list of ids to keep")
    ap.add_argument("--no-box", choices=["skip", "full"], default="skip",
                    help="what to do when a sample has no attention box. Default skip.")
    ap.add_argument("--save-viz", action="store_true", help="write a 2-panel check image per crop")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N kept records")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    crop_dir = os.path.join(args.out_dir, "crops")
    viz_dir = os.path.join(args.out_dir, "viz")
    os.makedirs(crop_dir, exist_ok=True)
    if args.save_viz:
        os.makedirs(viz_dir, exist_ok=True)

    records = _load_records(args.answers)
    keep_ids = _load_ids(args.ids_file)

    out = []
    stats = Counter()
    sides = []
    per_subset = defaultdict(Counter)
    errors = []

    for idx, rec in enumerate(records):
        rid = _record_id(rec, idx)
        subset = str(rec.get("subset") or rec.get("sub_dataset") or "unknown")
        if keep_ids is not None and rid not in keep_ids:
            stats["filtered"] += 1
            continue

        img_rel = _first_image(rec)
        img_path = _resolve_image(args.data_dir, img_rel)
        if img_path is None or not os.path.exists(img_path):
            stats["missing_image"] += 1
            errors.append({"id": rid, "reason": "missing_image", "image": img_rel})
            continue

        box = _extract_box(rec)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:  # pragma: no cover
            stats["open_error"] += 1
            errors.append({"id": rid, "reason": f"open_error:{e}", "image": img_rel})
            continue

        if box is None:
            if args.no_box == "skip":
                stats["no_box_skip"] += 1
                per_subset[subset]["no_box_skip"] += 1
                continue
            box = [0.0, 0.0, float(image.width), float(image.height)]
            stats["no_box_full"] += 1
            per_subset[subset]["no_box_full"] += 1

        try:
            box_orig = [float(v) for v in box[:4]]
            box_new = expand_box_eg_roi(box_orig, (image.width, image.height), min_side=args.min_side)
            x0, y0, x1, y1 = [int(round(v)) for v in box_new]
            x0, y0 = max(0, x0), max(0, y0)
            x1 = min(image.width, max(x0 + 1, x1))
            y1 = min(image.height, max(y0 + 1, y1))
            crop = image.crop((x0, y0, x1, y1))
        except Exception as e:  # pragma: no cover
            stats["crop_error"] += 1
            errors.append({"id": rid, "reason": f"crop_error:{e}", "box": box})
            continue

        crop_name = f"{_safe_name(subset, 40)}__{_safe_name(rid, 100)}.png"
        crop_path = os.path.join(crop_dir, crop_name)
        crop.save(crop_path)

        if args.save_viz:
            try:
                _save_viz(os.path.join(viz_dir, crop_name), image, box_orig, box_new)
            except Exception:
                pass

        side = max(x1 - x0, y1 - y0)
        sides.append(side)
        stats["cropped"] += 1
        per_subset[subset]["cropped"] += 1

        out.append({
            "id": rid,
            "subset": subset,
            "gt": rec.get("gt"),
            "image": os.path.abspath(crop_path),
            "origin_image": os.path.abspath(img_path),
            "origin_box": [round(float(v), 2) for v in box_orig],
            "egroi_box": [round(float(v), 2) for v in box_new],
            "crop_side": int(side),
        })

        if args.limit is not None and stats["cropped"] >= args.limit:
            break

    out_path = os.path.join(args.out_dir, "egroi_crops.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    report = {
        "answers": os.path.abspath(args.answers),
        "data_dir": os.path.abspath(args.data_dir),
        "min_side": args.min_side,
        "total_records": len(records),
        "stats": dict(stats),
        "side": {
            "min": min(sides) if sides else None,
            "max": max(sides) if sides else None,
            "mean": round(sum(sides) / len(sides), 2) if sides else None,
        },
        "per_subset": {k: dict(v) for k, v in sorted(per_subset.items())},
        "errors": errors[:50],
    }
    with open(os.path.join(args.out_dir, "crop_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(out)} crops -> {out_path}")
    print(f"next: python llava_eval/evaluate.py ... --data_path {os.path.abspath(args.out_dir)} "
          f"--question_file egroi_crops.json --text_src config:v0 --ab_tau 0.5")


if __name__ == "__main__":
    main()
