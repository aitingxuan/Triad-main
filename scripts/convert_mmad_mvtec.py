#!/usr/bin/env python3
"""
Convert MMAD mmad.json to Triad llava_eval format (MVTec-AD only, 0-shot).

Expected layout after downloading MMAD:
  data/MMAD/
    mmad.json
    MVTec-AD/
      bottle/
      cable/
      ...

Usage:
  python scripts/convert_mmad_mvtec.py --mmad_root data/MMAD
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llava_eval"))
from mmad_utils import (  # noqa: E402
    DEFAULT_IMAGE_ROOT,
    build_human_prompt,
    detect_image_root,
    is_mvtec_ad_image_path,
    iter_conversation_items,
    normalize_path,
    normalize_question_type,
    parse_mvtec_subset,
    resolve_existing_image_path,
    stable_shuffle_options,
)


def convert_record(
    image_path: str,
    record: dict,
    shuffle_options: bool,
    verify_images: bool,
    skip_missing_images: bool,
    mmad_root: str,
    image_root: str,
) -> List[dict]:
    subset = parse_mvtec_subset(image_path)
    if subset is None:
        return []

    rel_image, tried = resolve_existing_image_path(
        mmad_root, image_path, image_root=image_root
    )
    if rel_image is None:
        tried_lines = "\n  ".join(tried)
        msg = (
            f"image not found for mmad.json key: {image_path}\n"
            f"Tried under {mmad_root}:\n  {tried_lines}\n"
            "mmad.json uses DS-MVTec paths (…/image/…). Either:\n"
            "  1) extract DS-MVTec/ under mmad_root (recommended for MMAD), or\n"
            "  2) use official MVTec-AD/ (auto maps image/ -> test/)."
        )
        if skip_missing_images:
            print(f"WARNING: skip missing image -> {image_path}")
            return []
        raise FileNotFoundError(msg)

    samples: List[dict] = []
    for q_idx, qa in enumerate(iter_conversation_items(record)):
        question = qa.get("Question", "").strip()
        options = qa.get("Options", {})
        answer_key = qa.get("Answer")
        qtype = normalize_question_type(qa.get("type", "unknown"))
        if not question or not options or answer_key is None:
            continue

        seed_key = f"{rel_image}::{q_idx}::{question}"
        if shuffle_options:
            options, gt = stable_shuffle_options(options, answer_key, seed_key)
        else:
            options = {str(k): str(v) for k, v in options.items()}
            gt = str(answer_key)

        sample_id = f"{rel_image}__{qtype.replace(' ', '_')}__{q_idx}"
        human_value = build_human_prompt(question, options, zero_shot=True)
        samples.append(
            {
                "id": sample_id,
                "image": rel_image,
                "subset": subset,
                "question_type": qtype,
                "gt": gt,
                "options": options,
                "mmad_json_path": normalize_path(image_path),
                "origin_path": rel_image,
                "conversations": [
                    {"from": "human", "value": human_value},
                    {"from": "gpt", "value": ""},
                ],
            }
        )
    return samples


def load_mmad_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if raw.startswith("version https://git-lfs.github.com"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not real JSON.\n"
            "Run: cd data/MMAD && git lfs pull   OR download ALL_DATA.zip from Hugging Face."
        )
    return json.loads(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MMAD mmad.json to MVTec-AD 0-shot eval JSON.")
    parser.add_argument(
        "--mmad_root",
        type=str,
        default="data/MMAD",
        help="Root folder containing mmad.json and MVTec-AD/",
    )
    parser.add_argument(
        "--mmad_json",
        type=str,
        default="mmad.json",
        help="Annotation file relative to mmad_root (default: mmad.json)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="question_mvtec_0shot.json",
        help="Output JSON filename relative to mmad_root",
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Keep original option order from mmad.json",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=DEFAULT_IMAGE_ROOT,
        help="On-disk image folder preference: auto (default), DS-MVTec, or MVTec-AD. "
        "mmad.json DS-MVTec/.../image/... keys are resolved automatically.",
    )
    parser.add_argument(
        "--verify_images",
        action="store_true",
        help="Fail if any MVTec-AD image path is missing on disk",
    )
    parser.add_argument(
        "--skip_missing_images",
        action="store_true",
        help="Skip samples whose image file is missing (print warning)",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Debug: only convert first N MVTec-AD images",
    )
    args = parser.parse_args()

    mmad_root = os.path.abspath(args.mmad_root)
    if args.image_root == "auto":
        args.image_root = detect_image_root(mmad_root)
        print(f"Auto-detected image root: {args.image_root}/")
    mmad_json = os.path.join(mmad_root, args.mmad_json)
    out_path = os.path.join(mmad_root, args.out)

    if not os.path.isfile(mmad_json):
        raise FileNotFoundError(
            f"Missing {mmad_json}\n"
            "Download MMAD from https://huggingface.co/datasets/jiang-cc/MMAD "
            "and place mmad.json + MVTec-AD/ under data/MMAD/."
        )

    chat_ad = load_mmad_json(mmad_json)
    all_samples: List[dict] = []
    skipped = 0
    mvtec_images = 0
    json_root_stats: Dict[str, int] = {}

    for image_path, record in chat_ad.items():
        if not is_mvtec_ad_image_path(image_path):
            skipped += 1
            continue
        json_root = normalize_path(image_path).split("/")[0]
        json_root_stats[json_root] = json_root_stats.get(json_root, 0) + 1
        mvtec_images += 1
        if args.max_images is not None and mvtec_images > args.max_images:
            break
        all_samples.extend(
            convert_record(
                image_path,
                record,
                shuffle_options=not args.no_shuffle,
                verify_images=args.verify_images,
                skip_missing_images=args.skip_missing_images,
                mmad_root=mmad_root,
                image_root=args.image_root,
            )
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    qtypes: Dict[str, int] = {}
    subsets: Dict[str, int] = {}
    for s in all_samples:
        qtypes[s["question_type"]] = qtypes.get(s["question_type"], 0) + 1
        subsets[s["subset"]] = subsets.get(s["subset"], 0) + 1

    print(f"Wrote {len(all_samples)} questions from {mvtec_images} MVTec-AD images -> {out_path}")
    print(f"Skipped {skipped} non-MVTec entries (VisA / LOCO / GoodsAD, etc.)")
    print(f"mmad.json MVTec key prefixes: {dict(sorted(json_root_stats.items()))}")
    print(f"On-disk image root (preference): {args.image_root}/")
    print("Question types:", dict(sorted(qtypes.items())))
    print("Subsets:", dict(sorted(subsets.items())))
    if len(subsets) == 1:
        only = next(iter(subsets))
        print(
            f"WARNING: only subset '{only}' in output. "
            f"If unexpected, check mmad.json prefixes and {args.image_root}/ folder."
        )
    elif len(subsets) < 10:
        print(
            f"WARNING: only {len(subsets)} subsets in output (expected up to 15). "
            f"Missing classes may lack images under {args.image_root}/."
        )


if __name__ == "__main__":
    main()
