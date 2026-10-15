#!/usr/bin/env python3
"""Build PGRB juice interest masks from SAM+CLIP (LogSAD-style) segmentation.

Minimal layout under Seg_juice (one id-map PNG per image; meta only in index):

  Seg_juice/juice_interest_masks/
    classes.txt
    index.jsonl              # path, size, bbox/quality/usable per interest
    masks/<relative_id>.png  # uint8 id map at original resolution
    overlay/                 # only with --save_overlay

Interest ids: 0=bg, 1=juice, 2=top_label, 3=bottom_label, 4=bottle_body

Examples:
  # CoTM juice images (good + defects) — use this for PGRB train
  CUDA_VISIBLE_DEVICES=0 .../python -u scripts/build_process_interest_masks.py \\
    --source cotm_annotated --max_images -1

  # Raw MVTec-LOCO only (train has good only by dataset design)
  ... --source loco --split all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
_SEG = _REPO / "Seg_juice"
sys.path.insert(0, str(_SEG))

import interest_postprocess as ip  # noqa: E402
import sam_clip_juice_bottle as seg  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PGRB juice interest masks")
    p.add_argument(
        "--source",
        type=str,
        default="loco",
        choices=["loco", "cotm_annotated"],
        help="loco: MVTec-LOCO juice_bottle splits; cotm_annotated: CoTM training images (good+defects)",
    )
    p.add_argument("--dataset_path", type=str, default=seg._DEFAULT_DATASET)
    p.add_argument(
        "--cotm_image_root",
        type=str,
        default=str(_REPO / "movable" / "images" / "mvtec_loco_bottle_annotated"),
        help="Root with {banana,cherry,orange}_juice/image/<type>/*.png",
    )
    p.add_argument("--category", type=str, default="juice_bottle")
    p.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "validation", "all"],
        help="Only used when --source loco",
    )
    p.add_argument("--clip_cache_dir", type=str, default=seg._DEFAULT_CLIP_CACHE)
    p.add_argument("--sam_checkpoint", type=str, default=seg._DEFAULT_SAM_CKPT)
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--feat_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_images", type=int, default=-1)
    p.add_argument("--min_region", type=int, default=32)
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(_SEG / "juice_interest_masks"),
    )
    p.add_argument(
        "--save_overlay",
        action="store_true",
        help="Also write overlay/<stem>.png for visual QA",
    )
    return p.parse_args()


def collect_cotm_annotated(root: str) -> list[str]:
    root_p = Path(root)
    if not root_p.is_dir():
        raise FileNotFoundError(f"CoTM annotated root not found: {root_p}")
    paths: list[str] = []
    for flavor in ("banana_juice", "cherry_juice", "orange_juice"):
        img_root = root_p / flavor / "image"
        if not img_root.is_dir():
            continue
        paths.extend(sorted(str(p) for p in img_root.glob("*/*.png")))
    return paths


def relative_id_loco(image_path: str, dataset_path: str, category: str) -> str:
    cat_root = Path(dataset_path) / category
    rel = Path(image_path).resolve().relative_to(cat_root.resolve()).with_suffix("")
    return f"{category}/{rel.as_posix()}"


def relative_id_cotm(image_path: str, cotm_root: str) -> str:
    rel = Path(image_path).resolve().relative_to(Path(cotm_root).resolve()).with_suffix("")
    # banana_juice/image/good/002
    return rel.as_posix()


def stem_name(rel_id: str) -> str:
    return rel_id.replace("/", "__")


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    masks_dir = out_root / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_root / "overlay"
    if args.save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    ns = argparse.Namespace(
        device=args.device,
        image_size=args.image_size,
        clip_cache_dir=args.clip_cache_dir,
        sam_checkpoint=args.sam_checkpoint,
    )
    model_clip, tokenizer, mask_generator, transform, device = seg.build_models(ns)
    patch_query_obj = seg.encode_obj_text(
        model_clip, seg.DEFAULT_PATCH_QUERIES, tokenizer, device
    )

    image_paths = (
        collect_cotm_annotated(args.cotm_image_root)
        if args.source == "cotm_annotated"
        else seg.collect_images(args.dataset_path, args.category, args.split)
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No images for source={args.source} "
            f"(loco split={args.split} / cotm_root={args.cotm_image_root})"
        )
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    ori_feat_size = args.image_size // 14
    feature_list = [6, 12, 18, 24]

    with open(out_root / "classes.txt", "w", encoding="utf-8") as f:
        for name, idx in ip.INTEREST_ID.items():
            f.write(f"{idx}\t{name}\n")

    index_path = out_root / "index.jsonl"
    existing: dict[str, dict] = {}
    if index_path.is_file():
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                existing[row["relative_id"]] = row

    print(
        f"source={args.source} images={len(image_paths)} -> {out_root}",
        flush=True,
    )
    for path in tqdm(image_paths):
        if args.source == "cotm_annotated":
            rel_id = relative_id_cotm(path, args.cotm_image_root)
        else:
            rel_id = relative_id_loco(path, args.dataset_path, args.category)
        stem = stem_name(rel_id)
        result = seg.segment_one(
            path,
            model_clip,
            patch_query_obj,
            mask_generator,
            transform,
            device,
            args.image_size,
            args.feat_size,
            ori_feat_size,
            feature_list,
            args.min_region,
        )
        with Image.open(path) as im:
            ow, oh = im.size
        remapped = ip.remap_merge_to_interests(result["merge"], (oh, ow))

        mask_rel = f"masks/{stem}.png"
        ip.save_interest_id_png(out_root / mask_rel, remapped["interests_map"])

        if args.save_overlay:
            orig_rgb = np.array(Image.open(path).convert("RGB"))
            color = ip.colorize_interests(remapped["interests_map"])
            overlay = cv2.addWeighted(orig_rgb, 0.55, color, 0.45, 0)
            Image.fromarray(overlay).save(overlay_dir / f"{stem}.png")

        existing[rel_id] = {
            "relative_id": rel_id,
            "image_path": str(Path(path).resolve()),
            "mask": mask_rel,
            "size": remapped["size"],
            "interests": {
                k: {
                    "id": v["id"],
                    "bbox": v["bbox"],
                    "area": v["area"],
                    "quality": v["quality"],
                    "usable": v["usable"],
                }
                for k, v in remapped["interests"].items()
            },
        }

    with open(index_path, "w", encoding="utf-8") as f:
        for rid in sorted(existing.keys()):
            f.write(json.dumps(existing[rid], ensure_ascii=False) + "\n")

    print("Done.")
    print(f"Index: {index_path}")
    print("Per image: 1x masks/*.png (+ optional overlay). Meta only in index.jsonl.")


if __name__ == "__main__":
    main()
