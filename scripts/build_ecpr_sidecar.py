#!/usr/bin/env python3
"""Build ECPR sidecar from CoTM + offline interest masks (no SAM at build time).

Reads:
  - CoTM json (EG boxes already in sample['bbox'])
  - Seg_juice/juice_interest_masks_cotm/index.jsonl
  - movable/ecpr_trigger_rules.json

Writes:
  - movable/juice_ecpr_sidecar.jsonl
  - optional overlay/ for trigger QA

Example:
  python3 scripts/build_ecpr_sidecar.py
  python3 scripts/build_ecpr_sidecar.py --save_overlay --max_overlay 40
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[1]
_SLOTS = _REPO / "LLaVA-NeXT" / "llava" / "train" / "ecpr_slots.py"

# Load ecpr_slots directly — avoid `import llava` (pulls heavy model deps).
import importlib.util

_spec = importlib.util.spec_from_file_location("ecpr_slots", _SLOTS)
_ecpr = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_ecpr)
compose_for_sample = _ecpr.compose_for_sample
load_mask_index = _ecpr.load_mask_index
load_trigger_rules = _ecpr.load_trigger_rules
build_part_priors_from_good = _ecpr.build_part_priors_from_good
lookup_mask_row = _ecpr.lookup_mask_row
product_from_relative_id = _ecpr.product_from_relative_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ECPR juice sidecar")
    p.add_argument(
        "--cotm",
        type=str,
        default=str(_REPO / "movable" / "CoTM" / "CoTM_v6.json"),
    )
    p.add_argument(
        "--rules",
        type=str,
        default=str(_REPO / "movable" / "ecpr_trigger_rules.json"),
    )
    p.add_argument(
        "--mask_index",
        type=str,
        default=str(_REPO / "Seg_juice" / "juice_interest_masks_cotm" / "index.jsonl"),
    )
    p.add_argument(
        "--image_folder",
        type=str,
        default=str(_REPO / "movable" / "CoTM"),
        help="Prefix for relative CoTM image paths",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(_REPO / "movable" / "juice_ecpr_sidecar.jsonl"),
    )
    p.add_argument("--tau", type=float, default=None, help="Override rules iou_tau")
    p.add_argument(
        "--step_map",
        type=str,
        default=str(_REPO / "movable" / "process_step_map.json"),
        help="defect→Step fine map for Week-2 attr GT",
    )
    p.add_argument("--save_overlay", action="store_true")
    p.add_argument(
        "--overlay_dir",
        type=str,
        default=str(_REPO / "movable" / "juice_ecpr_overlay"),
    )
    p.add_argument("--max_overlay", type=int, default=60)
    return p.parse_args()


def _slot_positive_scores(
    boxes: list,
    slot_sources: list,
    slot_names: list,
    defect_type: Optional[str],
    gt_interests: list,
    mask_row: Optional[dict],
    part_priors_product: Optional[dict],
) -> list:
    """Per-slot soft target in [0,1]: IoU vs GT interest bbox (or 1.0 for missing prior)."""
    box_iou = _ecpr.box_iou
    interests = (mask_row or {}).get("interests") or {}
    out = []
    is_good = defect_type in (None, "good")
    for box, src, name in zip(boxes, slot_sources, slot_names):
        if is_good:
            out.append(0.0)
            continue
        # Prefer named interest overlap when this slot is a part.
        target_names = []
        if src == "part" and name in gt_interests:
            target_names = [name]
        elif src == "part":
            target_names = [name] if name else list(gt_interests)
        else:
            target_names = list(gt_interests)
        best = 0.0
        for tn in target_names:
            meta = interests.get(tn) or {}
            bbox = None
            if meta.get("usable") and meta.get("bbox") and len(meta["bbox"]) >= 4:
                bbox = meta["bbox"][:4]
            elif part_priors_product and tn in part_priors_product:
                bbox = part_priors_product[tn][:4]
            if bbox is None:
                # missing_* part slot with prior already in ``box``: treat as positive evidence slot
                if src == "part" and tn in gt_interests and "missing" in str(defect_type or ""):
                    best = max(best, 1.0)
                continue
            best = max(best, float(box_iou(box, bbox)))
        out.append(float(best))
    return out


def _gt_from_step_map(step_map: dict, product: Optional[str], defect_type: Optional[str]) -> dict:
    if not product or not defect_type or defect_type == "good":
        return {
            "gt_step": None,
            "gt_step_name": None,
            "gt_type_text": None,
            "gt_location_tag": None,
        }
    prod = step_map.get(product) or {}
    defects = prod.get("defects") or {}
    info = defects.get(defect_type) or {}
    return {
        "gt_step": info.get("step"),
        "gt_step_name": info.get("step_name"),
        "gt_type_text": info.get("type_text"),
        "gt_location_tag": info.get("location"),
    }


def _draw_overlay(image_path: Path, row: dict, out_path: Path) -> None:
    from PIL import Image, ImageDraw

    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    # EG original
    for b in row.get("eg_boxes") or []:
        draw.rectangle(b, outline=(0, 180, 255), width=4)
    # final slots
    for b, src, name in zip(row["boxes"], row["slot_sources"], row["slot_names"]):
        color = (255, 64, 64) if src == "part" else (0, 220, 80)
        draw.rectangle(b, outline=color, width=3)
        draw.text((b[0] + 4, b[1] + 4), f"{src}:{name}", fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def main() -> None:
    args = parse_args()
    rules = load_trigger_rules(args.rules)
    mask_index = load_mask_index(args.mask_index)
    products = set(rules.get("products") or [])
    part_priors = build_part_priors_from_good(mask_index, products=sorted(products) or None)
    with open(args.step_map, "r") as f:
        step_map = json.load(f)
    defect_to_interests = rules.get("defect_to_interests") or {}
    print(
        "part_priors",
        {p: list(v.keys()) for p, v in part_priors.items()},
    )
    with open(args.cotm, "r") as f:
        samples = json.load(f)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    trigger_by_defect = Counter()
    n_overlay = 0

    with open(out_path, "w") as fout:
        for sample in samples:
            img_rel = sample.get("image") or sample.get("src")
            if not img_rel:
                continue
            sub = sample.get("sub_dataset") or ""
            defect = sample.get("defect_type")
            # only juice CoTM with closed-set products
            is_juice = any(p in str(img_rel) or p == sub for p in products)
            if not is_juice:
                stats["skip_non_juice"] += 1
                continue

            abs_img = Path(args.image_folder) / img_rel
            eg_raw = sample.get("bbox")
            # CoTM stores bbox as [boxes_for_img0] for single image
            if isinstance(eg_raw, list) and len(eg_raw) == 1 and eg_raw and isinstance(eg_raw[0], list):
                # either [[box,...]] or [[[x..],[x..]]]
                if eg_raw[0] and isinstance(eg_raw[0][0], (list, tuple)):
                    eg_for_compose = eg_raw[0]
                else:
                    eg_for_compose = eg_raw
            else:
                eg_for_compose = eg_raw

            image_path = str(abs_img if abs_img.exists() else img_rel)
            composed = compose_for_sample(
                image_path=image_path,
                defect_type=defect,
                eg_boxes_raw=eg_for_compose,
                rules=rules,
                mask_index=mask_index,
                tau=args.tau,
                part_priors=part_priors,
            )
            stats["juice"] += 1
            if composed["has_mask"]:
                stats["with_mask"] += 1
            else:
                stats["no_mask"] += 1
            if composed["n_part"] > 0:
                stats["injected_part"] += 1
                trigger_by_defect[str(defect)] += 1
            else:
                stats["eg_only"] += 1
            if composed["replaced"]:
                stats["replaced_eg"] += 1

            rid = composed["relative_id"]
            product = product_from_relative_id(rid) or (
                sub if str(sub).endswith("_juice") else None
            )
            gt_interests = list(defect_to_interests.get(defect) or [])
            gt_fields = _gt_from_step_map(step_map, product, defect)
            mask_row = lookup_mask_row(mask_index, image_path=image_path, relative_id=rid)
            slot_positive = _slot_positive_scores(
                composed["boxes"],
                composed["slot_sources"],
                composed["slot_names"],
                defect,
                gt_interests,
                mask_row,
                part_priors.get(product or ""),
            )
            is_anomaly = defect not in (None, "good")
            if is_anomaly and any(x > 0 for x in slot_positive):
                stats["slot_pos_hit"] += 1

            row = {
                "id": sample.get("id"),
                "image": img_rel,
                "image_path": str(abs_img),
                "sub_dataset": sub,
                "defect_type": defect,
                "relative_id": rid,
                "has_mask": composed["has_mask"],
                "eg_boxes": composed["eg_boxes"],
                "triggered_parts": composed["triggered_parts"],
                "part_sources": composed.get("part_sources"),
                "boxes": composed["boxes"],
                "slot_sources": composed["slot_sources"],
                "slot_names": composed["slot_names"],
                "replaced": composed["replaced"],
                "iou_tau": composed["iou_tau"],
                "n_eg": composed.get("n_eg"),
                "n_part": composed.get("n_part"),
                # Week-2 supervision
                "is_anomaly": is_anomaly,
                "gt_interests": gt_interests,
                "slot_positive": slot_positive,
                **gt_fields,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            if args.save_overlay and composed["n_part"] > 0 and n_overlay < args.max_overlay:
                if abs_img.exists():
                    safe = (composed["relative_id"] or f"sample_{stats['juice']}").replace("/", "__")
                    _draw_overlay(abs_img, row, Path(args.overlay_dir) / f"{safe}.png")
                    n_overlay += 1

    print("Wrote", out_path)
    print("stats", dict(stats))
    print("injected_by_defect", dict(trigger_by_defect.most_common()))
    if args.save_overlay:
        print("overlays", n_overlay, "->", args.overlay_dir)


if __name__ == "__main__":
    main()
