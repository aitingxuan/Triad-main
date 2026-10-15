#!/usr/bin/env python3
"""A1 + A3 small-sample causal probe (standalone; does not modify existing eval code).

A1 — visual evidence interventions on the image:
  - gt_occlude: fill GT defect pixels (mean color)
  - bg_occlude: fill an equal-area background box (control)

A3 — ROI source interventions (image unchanged):
  - roi_musc: MusC heatmap → ROI (default eval path)
  - roi_gt:   GT mask → ROI boxes
  - roi_random: random boxes away from defect
  - roi_none: empty ROI list (anyres only)
  - roi_adversarial: box on normal region (away from GT)

Baseline condition name: ``baseline`` (= original image + MusC ROI).

Example:
  conda activate triad
  cd /home/yz/code/Triad-main
  CUDA_VISIBLE_DEVICES=0 python llava_eval/causal_a1_a3_probe.py \\
    --model-path checkpoints/merged_llava_qwen_triad_9ROI_cotmv3_vis_ori \\
    --num-samples 8 \\
    --text_src config:cotm_v3 \\
    --max_new_tokens 512 \\
    --out LLM_judge_output/causal_a1_a3_smoke.json
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Any, Dict, List, Optional

from PIL import Image
from tqdm import tqdm

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from causal_probe_common import (  # noqa: E402
    infer_subset,
    load_binary_mask,
    load_prompt_config,
    load_worker,
    mask_bbox,
    musc_to_binary,
    occlude_bbox_region,
    occlude_with_mask,
    random_bbox_away_from_mask,
    resolve_prompt,
    run_once,
    save_json,
    square_expand_bbox,
    stratified_pick,
    summarize_condition_rows,
)
from evaluate import data_read, read_mask  # noqa: E402


def _load_image(data_dir: str, sample: dict) -> Image.Image:
    rel = sample.get("image")
    if isinstance(rel, list):
        rel = rel[0]
    path = rel if os.path.isabs(rel) else os.path.join(data_dir, rel)
    return Image.open(path).convert("RGB")


def _load_musc(data_dir: str, sample: dict, img: Image.Image):
    if not sample.get("mask"):
        return None, None
    rel = sample["mask"]
    if isinstance(rel, list):
        rel = rel[0]
    path = rel if os.path.isabs(rel) else os.path.join(data_dir, rel)
    raw = read_mask(path)
    binary = musc_to_binary(raw, img.size)
    return raw, binary


def build_conditions(
    img: Image.Image,
    sample: dict,
    musc_raw,
    musc_bin,
    gt_mask,
    rng: random.Random,
    enable_a1: bool,
    enable_a3: bool,
) -> List[Dict[str, Any]]:
    """Each condition: name, image, masks, boxes_list, meta."""
    w, h = img.size
    conds: List[Dict[str, Any]] = []

    # --- baseline: original + MusC ROI (masks path, same as evaluate default) ---
    conds.append(
        {
            "name": "baseline",
            "image": img,
            "masks": [musc_raw] if musc_raw is not None else None,
            "boxes_list": None,
            "meta": {"group": "baseline"},
        }
    )

    side = 128
    gt_bb = mask_bbox(gt_mask) if gt_mask is not None else None
    if gt_bb is not None:
        side = max(gt_bb[2] - gt_bb[0] + 1, gt_bb[3] - gt_bb[1] + 1, 64)
        side = int(min(side, w, h))

    if enable_a1 and gt_mask is not None and gt_mask.any():
        occl = occlude_with_mask(img, gt_mask, mode="mean")
        conds.append(
            {
                "name": "gt_occlude",
                "image": occl,
                "masks": [musc_raw] if musc_raw is not None else None,
                "boxes_list": None,
                "meta": {"group": "A1", "note": "GT defect pixels filled with mean color; ROI still MusC"},
            }
        )
        bg_box = random_bbox_away_from_mask(gt_mask, side, rng)
        bg_img = occlude_bbox_region(img, bg_box, mode="mean")
        conds.append(
            {
                "name": "bg_occlude",
                "image": bg_img,
                "masks": [musc_raw] if musc_raw is not None else None,
                "boxes_list": None,
                "meta": {"group": "A1", "bg_box": bg_box, "note": "equal-ish area background occlusion control"},
            }
        )

    if enable_a3:
        # MusC ROI explicit (same as baseline, kept for clarity in summaries)
        conds.append(
            {
                "name": "roi_musc",
                "image": img,
                "masks": [musc_raw] if musc_raw is not None else None,
                "boxes_list": None,
                "meta": {"group": "A3"},
            }
        )
        if gt_mask is not None and gt_mask.any():
            conds.append(
                {
                    "name": "roi_gt",
                    "image": img,
                    "masks": [gt_mask.astype("float32")],
                    "boxes_list": None,
                    "meta": {"group": "A3", "note": "ROI from GT mask contours"},
                }
            )
            adv = random_bbox_away_from_mask(gt_mask, min(side, 384), rng)
            sq = square_expand_bbox(adv, w, h, side=min(max(side, 128), 384))
            conds.append(
                {
                    "name": "roi_adversarial",
                    "image": img,
                    "masks": None,
                    "boxes_list": [[sq]],
                    "meta": {"group": "A3", "box": sq},
                }
            )
            rnd = random_bbox_away_from_mask(gt_mask, min(side, 384), rng)
            rnd = square_expand_bbox(rnd, w, h, side=min(max(side, 128), 384))
            conds.append(
                {
                    "name": "roi_random",
                    "image": img,
                    "masks": None,
                    "boxes_list": [[rnd]],
                    "meta": {"group": "A3", "box": rnd},
                }
            )
        else:
            # good sample: random / none still meaningful
            rnd = [rng.randint(0, max(w - 128, 0)), rng.randint(0, max(h - 128, 0))]
            rnd = [rnd[0], rnd[1], rnd[0] + 127, rnd[1] + 127]
            conds.append(
                {
                    "name": "roi_random",
                    "image": img,
                    "masks": None,
                    "boxes_list": [[rnd]],
                    "meta": {"group": "A3", "box": rnd},
                }
            )

        conds.append(
            {
                "name": "roi_none",
                "image": img,
                "masks": None,
                "boxes_list": [[]],  # empty extra ROI patches
                "meta": {"group": "A3", "note": "no ROI crops; anyres global only"},
            }
        )

    return conds


def main():
    parser = argparse.ArgumentParser(description="A1+A3 causal probe (small sample)")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="evaluation/mvtec")
    parser.add_argument("--ds", type=str, default="mvtec")
    parser.add_argument("--from", dest="question_file", type=str, default="question_musc.jsonl")
    parser.add_argument("--text_src", type=str, default="config:cotm_v3")
    parser.add_argument("--mvtec-root", type=str, default="data/MMAD/MVTec-AD")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--no-a1", action="store_true", help="Skip image occlusion conditions")
    parser.add_argument("--no-a3", action="store_true", help="Skip ROI triangle conditions")
    parser.add_argument(
        "--include-good",
        action="store_true",
        help="Also sample good images (A1 skipped when no GT mask)",
    )
    parser.add_argument("--save-images-dir", type=str, default=None, help="Optional dir to dump intervened images")
    parser.add_argument("--out", type=str, default="LLM_judge_output/causal_a1_a3_probe.json")
    args = parser.parse_args()

    data_dir = args.data_path
    if data_dir.endswith("/") and not data_dir.rstrip("/").endswith(args.ds):
        data_dir = os.path.join(data_dir, args.ds)

    qs_tmp, prompt_ver = load_prompt_config(args.ds, args.text_src)
    question_path = os.path.join(data_dir, args.question_file)
    all_datas = data_read(question_path)

    picked = stratified_pick(
        all_datas,
        num_samples=args.num_samples,
        seed=args.seed,
        require_gt_mask=not args.include_good,
        mvtec_root=args.mvtec_root,
        defect_only=not args.include_good,
    )
    if not picked:
        raise RuntimeError(
            "No samples picked. Check --mvtec-root GT masks and question file. "
            "For good-only debug use --include-good (A1 limited)."
        )

    print(f"picked {len(picked)} samples | prompt={prompt_ver} | A1={not args.no_a1} A3={not args.no_a3}")
    worker = load_worker(args.model_path, args.model_base, args.use_flash_attn)
    rng = random.Random(args.seed)

    rows: List[dict] = []
    for sample in tqdm(picked, desc="A1+A3"):
        sid = str(sample.get("id") or sample.get("question_id"))
        subset = infer_subset(sample)
        gt = int(sample.get("gt", 0))
        prompt = resolve_prompt(qs_tmp, subset)
        img = _load_image(data_dir, sample)
        musc_raw, musc_bin = _load_musc(data_dir, sample, img)

        gt_mask = None
        gt_path = sample.get("_gt_mask_path") or None
        if gt_path is None and gt == 1:
            from causal_probe_common import resolve_gt_mask_path

            gt_path = resolve_gt_mask_path(sample.get("origin_path", ""), args.mvtec_root)
        if gt_path:
            gt_mask = load_binary_mask(gt_path, img.size)

        conds = build_conditions(
            img,
            sample,
            musc_raw,
            musc_bin,
            gt_mask,
            rng,
            enable_a1=not args.no_a1,
            enable_a3=not args.no_a3,
        )

        for cond in conds:
            if args.save_images_dir:
                od = os.path.join(args.save_images_dir, sid)
                os.makedirs(od, exist_ok=True)
                cond["image"].save(os.path.join(od, f"{cond['name']}.png"))

            parsed = run_once(
                worker,
                prompt,
                cond["image"],
                cond["masks"],
                cond["boxes_list"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            rows.append(
                {
                    "id": sid,
                    "subset": subset,
                    "gt": gt,
                    "origin_path": sample.get("origin_path"),
                    "gt_mask_path": gt_path,
                    "condition": cond["name"],
                    "group": cond["meta"].get("group"),
                    "condition_meta": {k: v for k, v in cond["meta"].items() if k != "group"},
                    "answer": parsed["answer"],
                    "location": parsed["location"],
                    "type": parsed["type"],
                    "think_head": (parsed["think"] or "")[:240],
                    "raw": parsed["raw"],
                    "correct": (parsed["answer"] == gt) if parsed["answer"] is not None else None,
                }
            )

    summary = summarize_condition_rows(rows)
    # A1-specific: sensitivity = flip(gt_occlude) - flip(bg_occlude)
    a1_note = {}
    if "gt_occlude" in summary and "bg_occlude" in summary:
        a1_note = {
            "gt_occlude_flip_rate": summary["gt_occlude"]["answer_flip_rate"],
            "bg_occlude_flip_rate": summary["bg_occlude"]["answer_flip_rate"],
            "causal_gap_gt_minus_bg": summary["gt_occlude"]["answer_flip_rate"]
            - summary["bg_occlude"]["answer_flip_rate"],
            "interpretation": (
                "If gt_occlude flips >> bg_occlude, decisions depend on defect pixels; "
                "if both ~0, model likely ignores local visual evidence."
            ),
        }
    a3_note = {
        "compare": ["roi_gt", "roi_musc", "roi_random", "roi_adversarial", "roi_none"],
        "interpretation": (
            "If roi_gt >> roi_random/adversarial and roi_none drops, ROI pathway is used; "
            "if all similar to baseline, ROI tokens are weakly causal."
        ),
    }

    report = {
        "meta": {
            "experiment": "A1_visual_occlusion + A3_roi_triangle",
            "model_path": args.model_path,
            "data_path": data_dir,
            "question_file": args.question_file,
            "text_src": args.text_src,
            "prompt_version": prompt_ver,
            "mvtec_root": args.mvtec_root,
            "num_samples": len(picked),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "enable_a1": not args.no_a1,
            "enable_a3": not args.no_a3,
        },
        "summary": summary,
        "a1_causal_readout": a1_note,
        "a3_causal_readout": a3_note,
        "per_run": rows,
    }
    save_json(report, args.out)
    print("=== summary (acc / flip_vs_baseline) ===")
    for k, v in summary.items():
        if k.startswith("_"):
            continue
        print(
            f"  {k:18s} n={v['n']:3d} acc={v['acc']} "
            f"flip={v['answer_flip_rate']:.3f} yes_rate={v['yes_rate']}"
        )
    if a1_note:
        print("A1 causal_gap_gt_minus_bg =", round(a1_note["causal_gap_gt_minus_bg"], 4))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
