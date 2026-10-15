#!/usr/bin/env python3
"""Standalone v0 A/B binary-softmax confidence probe (does not modify existing eval).

For each musc_v0-style sample:
  1) Build the same multimodal prompt as evaluate.py / causal probes
  2) Greedy decode first token with output_scores=True
  3) Two-way softmax over choice letters A(=Yes/defect) and B(=No/normal)
  4) Aggregate mean confidence on TP / FP / TN / FN

Example (smoke):
  conda activate triad
  cd /home/yz/code/Triad-main
  CUDA_VISIBLE_DEVICES=0 python llava_eval/v0_softmax_confidence_probe.py \\
    --model-path checkpoints/merged_llava_qwen_triad_9ROI_cotmv6_ecprloss \\
    --num-samples 16 \\
    --text_src config:v0 \\
    --out LLM_judge_output/v0_softmax_conf_smoke.json

Full MVTec musc_v0:
  CUDA_VISIBLE_DEVICES=0 python llava_eval/v0_softmax_confidence_probe.py \\
    --model-path checkpoints/merged_llava_qwen_triad_9ROI_cotmv6_ecprloss \\
    --text_src config:v0 \\
    --out LLM_judge_output/v0_softmax_conf_ecprloss.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
_LLAVA = os.path.join(_REPO, "LLaVA-NeXT")
if _LLAVA not in sys.path:
    sys.path.insert(0, _LLAVA)

from llava.constants import (  # noqa: E402
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: E402

from causal_probe_common import (  # noqa: E402
    infer_subset,
    load_prompt_config,
    load_worker,
    resolve_prompt,
    stratified_pick,
)
from evaluate import data_read, read_mask  # noqa: E402

CELL_ORDER = ("TP", "FP", "TN", "FN", "UN")


# ---------------------------------------------------------------------------
# Token / math helpers
# ---------------------------------------------------------------------------

def _single_token_ids(tokenizer, texts: Sequence[str]) -> List[int]:
    ids: List[int] = []
    seen = set()
    unk = getattr(tokenizer, "unk_token_id", None)
    for text in texts:
        enc = tokenizer.encode(text, add_special_tokens=False)
        if len(enc) == 1:
            tid = int(enc[0])
            if tid not in seen and tid != unk:
                seen.add(tid)
                ids.append(tid)
        tid2 = tokenizer.convert_tokens_to_ids(text)
        if isinstance(tid2, int) and tid2 != unk and tid2 not in seen:
            # convert_tokens_to_ids may return unk for multi-char; skip junk
            if tokenizer.convert_ids_to_tokens(tid2) is not None:
                seen.add(tid2)
                ids.append(tid2)
    return ids


def resolve_ab_token_ids(tokenizer) -> Dict[str, List[int]]:
    """Collect single-token surface forms for A/B (with/without leading space)."""
    return {
        "A": _single_token_ids(tokenizer, ["A", " A", "A.", " A."]),
        "B": _single_token_ids(tokenizer, ["B", " B", "B.", " B."]),
    }


def two_way_softmax(z_a: float, z_b: float) -> Tuple[float, float]:
    m = max(z_a, z_b)
    ea = math.exp(z_a - m)
    eb = math.exp(z_b - m)
    s = ea + eb
    return ea / s, eb / s


def confusion_cell(gt: int, pred: Optional[int]) -> str:
    """gt/pred: 1=anomaly(A), 0=normal(B)."""
    if pred is None:
        return "UN"
    if gt == 1 and pred == 1:
        return "TP"
    if gt == 0 and pred == 1:
        return "FP"
    if gt == 0 and pred == 0:
        return "TN"
    if gt == 1 and pred == 0:
        return "FN"
    return "UN"


def _stat(xs: Sequence[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# ---------------------------------------------------------------------------
# Prompt / image IO (mirror evaluate + causal probes; no edits there)
# ---------------------------------------------------------------------------

def build_chat_prompt(worker, user_text: str, image_num: int = 1) -> str:
    state = worker.new_state()
    text = user_text
    if DEFAULT_IMAGE_TOKEN not in text:
        text = DEFAULT_IMAGE_TOKEN + "\n" + text
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    prompt = state.get_prompt()
    if prompt.count(DEFAULT_IMAGE_TOKEN) == 1 and image_num != 1:
        img_s = "\n".join([DEFAULT_IMAGE_TOKEN] * image_num)
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, img_s)
    return prompt


def load_sample_image(data_dir: str, sample: dict) -> Image.Image:
    rel = sample.get("image")
    if isinstance(rel, list):
        rel = rel[0]
    path = rel if os.path.isabs(str(rel)) else os.path.join(data_dir, rel)
    return Image.open(path).convert("RGB")


def load_sample_mask(data_dir: str, sample: dict):
    if not sample.get("mask"):
        return None
    rel = sample["mask"]
    if isinstance(rel, list):
        rel = rel[0]
    path = rel if os.path.isabs(str(rel)) else os.path.join(data_dir, rel)
    return read_mask(path)


# ---------------------------------------------------------------------------
# Core: first-step A/B logits
# ---------------------------------------------------------------------------

@torch.inference_mode()
def score_ab_first_token(
    worker,
    prompt: str,
    images: List[Image.Image],
    masks: Optional[List[Any]] = None,
    boxes_list: Optional[List[Any]] = None,
    ab_ids: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, Any]:
    """Greedy 1-token generate with scores; return A/B two-way softmax confidences.

    Mirrors SimpleModelWorker.generate_batch preprocessing, but does not change that file.
    """
    sm = worker.model
    tokenizer, model = sm.tokenizer, sm.model
    if ab_ids is None:
        ab_ids = resolve_ab_token_ids(tokenizer)

    image_args: Dict[str, Any] = {}
    num_image_tokens = 0
    prompt_use = prompt
    if images is not None and len(images) > 0 and sm.is_multimodal:
        if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
            raise ValueError("Number of images does not match number of <image> tokens in prompt")
        image_sizes = [image.size for image in images]
        images_t, roi_bboxes = process_images(
            images, sm.image_processor, model.config, masks=masks, boxes_list=boxes_list, return_bboxes=True
        )
        if type(images_t) is list:
            if type(images_t[0]) is not dict:
                images_t = [im.to(model.device, dtype=model.dtype) for im in images_t]
        else:
            images_t = images_t.to(model.device, dtype=model.dtype)
        replace_token = DEFAULT_IMAGE_TOKEN
        if getattr(model.config, "mm_use_im_start_end", False):
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        prompt_use = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        num_image_tokens = prompt_use.count(replace_token) * model.get_vision_tower().num_patches
        image_args = {"images": images_t, "image_sizes": image_sizes, "roi_bboxes": roi_bboxes}

    input_ids = tokenizer_image_token(prompt_use, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(
        sm.device
    )
    max_context_length = getattr(model.config, "max_position_embeddings", 2048)
    max_new_tokens = min(1, max_context_length - input_ids.shape[-1] - num_image_tokens)
    if max_new_tokens < 1:
        return {"error": "context_overflow", "pred": None, "p_A": None, "p_B": None}

    out = model.generate(
        inputs=input_ids,
        do_sample=False,
        max_new_tokens=1,
        use_cache=True,
        output_scores=True,
        return_dict_in_generate=True,
        **image_args,
    )

    # scores[0]: [B, V] logits of the first generated step
    logits = out.scores[0][0].float().cpu()
    seq = out.sequences[0]
    # With inputs_embeds, sequences are usually generated tokens only
    gen_tid = int(seq[-1].item())
    gen_tok = tokenizer.convert_ids_to_tokens(gen_tid)
    raw_text = tokenizer.decode([gen_tid], skip_special_tokens=True)

    a_ids = ab_ids["A"]
    b_ids = ab_ids["B"]
    if not a_ids or not b_ids:
        raise RuntimeError(f"Failed to resolve A/B token ids: {ab_ids}")

    z_a = float(max(logits[i].item() for i in a_ids))
    z_b = float(max(logits[i].item() for i in b_ids))
    p_a, p_b = two_way_softmax(z_a, z_b)

    # Full-vocab softmax mass on chosen surface forms (diagnostic; usually tiny)
    full = torch.softmax(logits, dim=-1)
    p_a_full = float(sum(full[i].item() for i in a_ids))
    p_b_full = float(sum(full[i].item() for i in b_ids))

    # Hard pred: argmax over A vs B two-way (aligned with p_A)
    pred = 1 if p_a >= p_b else 0
    # Also record whether greedy token is literally A/B
    greedy_letter = None
    if gen_tid in a_ids:
        greedy_letter = "A"
    elif gen_tid in b_ids:
        greedy_letter = "B"
    else:
        t = (raw_text or "").strip().upper()
        if t.startswith("A"):
            greedy_letter = "A"
        elif t.startswith("B"):
            greedy_letter = "B"

    return {
        "pred": pred,  # 1=A/Yes, 0=B/No
        "pred_letter": "A" if pred == 1 else "B",
        "p_A": p_a,
        "p_B": p_b,
        "z_A": z_a,
        "z_B": z_b,
        "margin": z_a - z_b,
        "p_A_fullvocab": p_a_full,
        "p_B_fullvocab": p_b_full,
        "greedy_token_id": gen_tid,
        "greedy_token": gen_tok,
        "greedy_text": raw_text,
        "greedy_letter": greedy_letter,
        "ab_token_ids": ab_ids,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_rows(rows: List[dict]) -> Dict[str, Any]:
    """Build confusion counts + confidence stats for TP/FP/TN/FN (and overall metrics)."""

    def pack(group_rows: List[dict]) -> Dict[str, Any]:
        cells: Dict[str, List[dict]] = {c: [] for c in CELL_ORDER}
        for r in group_rows:
            cells[r["cell"]].append(r)

        cell_stats = {}
        for c in CELL_ORDER:
            rs = cells[c]
            # p_yes = P(A); p_pred = confidence of the chosen letter
            p_yes = [r["p_A"] for r in rs if r.get("p_A") is not None]
            p_pred = []
            for r in rs:
                if r.get("p_A") is None:
                    continue
                p_pred.append(r["p_A"] if r["pred"] == 1 else r["p_B"])
            margins = [r["margin"] for r in rs if r.get("margin") is not None]
            cell_stats[c] = {
                "n": len(rs),
                "p_A": _stat(p_yes),
                "p_pred": _stat(p_pred),
                "margin_A_minus_B": _stat(margins),
            }

        tp, fp, tn, fn, un = (cell_stats[c]["n"] for c in CELL_ORDER)
        eps = 1e-15
        decided = tp + fp + tn + fn
        acc = (tp + tn) / (decided + un + eps)
        pre = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        tnr = tn / (tn + fp + eps)
        return {
            "counts": {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "UN": un, "N": len(group_rows)},
            "metrics": {"acc": acc, "pre": pre, "rec": rec, "tnr": tnr},
            "cells": cell_stats,
        }

    by_subset: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_subset[r["subset"]].append(r)

    per_subset = {k: pack(v) for k, v in sorted(by_subset.items())}
    overall = pack(rows)

    # macro mean of per-subset metrics (same spirit as collect_acc mean row)
    if per_subset:
        macro = {
            m: float(np.mean([per_subset[s]["metrics"][m] for s in per_subset]))
            for m in ("acc", "pre", "rec", "tnr")
        }
    else:
        macro = {"acc": None, "pre": None, "rec": None, "tnr": None}

    # macro mean of cell mean p_A
    macro_cells = {}
    for c in CELL_ORDER:
        means = []
        for s in per_subset:
            st = per_subset[s]["cells"][c]["p_A"]
            if st["n"] > 0 and st["mean"] is not None:
                means.append(st["mean"])
        macro_cells[c] = {
            "subsets_with_samples": len(means),
            "mean_of_subset_mean_p_A": float(np.mean(means)) if means else None,
        }

    return {
        "all": overall,
        "mean_metrics": macro,
        "mean_cell_p_A": macro_cells,
        "per_subset": per_subset,
    }


def summary_to_csv_rows(summary: Dict[str, Any]) -> List[List[Any]]:
    """Flat CSV: one row per subset (+ mean/all), with metrics and cell mean p_A / p_pred."""
    header = [
        "subset",
        "acc",
        "pre",
        "rec",
        "tnr",
        "TP",
        "FP",
        "TN",
        "FN",
        "UN",
        "mean_pA_TP",
        "mean_pA_FP",
        "mean_pA_TN",
        "mean_pA_FN",
        "mean_pPred_TP",
        "mean_pPred_FP",
        "mean_pPred_TN",
        "mean_pPred_FN",
    ]
    rows_out = [header]

    def one(name: str, block: Dict[str, Any]) -> List[Any]:
        m = block["metrics"]
        c = block["counts"]
        cells = block["cells"]

        def pa(cell: str):
            return cells[cell]["p_A"]["mean"]

        def pp(cell: str):
            return cells[cell]["p_pred"]["mean"]

        return [
            name,
            m["acc"],
            m["pre"],
            m["rec"],
            m["tnr"],
            c["TP"],
            c["FP"],
            c["TN"],
            c["FN"],
            c["UN"],
            pa("TP"),
            pa("FP"),
            pa("TN"),
            pa("FN"),
            pp("TP"),
            pp("FP"),
            pp("TN"),
            pp("FN"),
        ]

    for subset, block in summary["per_subset"].items():
        rows_out.append(one(subset, block))

    # mean row: macro metrics; cell means = mean of subset means
    mm = summary["mean_metrics"]
    mc = summary["mean_cell_p_A"]
    rows_out.append(
        [
            "mean",
            mm["acc"],
            mm["pre"],
            mm["rec"],
            mm["tnr"],
            "",
            "",
            "",
            "",
            "",
            mc["TP"]["mean_of_subset_mean_p_A"],
            mc["FP"]["mean_of_subset_mean_p_A"],
            mc["TN"]["mean_of_subset_mean_p_A"],
            mc["FN"]["mean_of_subset_mean_p_A"],
            "",
            "",
            "",
            "",
        ]
    )
    rows_out.append(one("all", summary["all"]))
    return rows_out


def print_summary(summary: Dict[str, Any]) -> None:
    all_b = summary["all"]
    print("=== overall metrics ===")
    print(
        f"acc={all_b['metrics']['acc']:.4f}  pre={all_b['metrics']['pre']:.4f}  "
        f"rec={all_b['metrics']['rec']:.4f}  tnr={all_b['metrics']['tnr']:.4f}"
    )
    print("counts:", all_b["counts"])
    print("=== cell mean P(A)=P(Yes) / mean P(pred) ===")
    for c in ("TP", "FP", "TN", "FN"):
        cs = all_b["cells"][c]
        pa = cs["p_A"]["mean"]
        pp = cs["p_pred"]["mean"]
        pa_s = f"{pa:.4f}" if pa is not None else "n/a"
        pp_s = f"{pp:.4f}" if pp is not None else "n/a"
        print(f"  {c}: n={cs['n']:4d}  mean_p_A={pa_s}  mean_p_pred={pp_s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="v0 A/B binary-softmax confidence probe")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--ds", type=str, default="mvtec")
    parser.add_argument("--data_path", type=str, default="./evaluation/mvtec")
    parser.add_argument("--from", dest="question_file", type=str, default="question_musc.jsonl")
    parser.add_argument("--text_src", type=str, default="config:v0")
    parser.add_argument("--subset", type=str, default="all", help="all or comma-separated class names")
    parser.add_argument("--num-samples", type=int, default=-1, help="-1 = all; else stratified subset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True, help="JSON output path")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV summary path (default: alongside --out)")
    parser.add_argument("--no-rows", action="store_true", help="Omit per-sample rows from JSON (summary only)")
    args = parser.parse_args()

    data_dir = args.data_path
    if data_dir.endswith("/") and not data_dir.rstrip("/").endswith(args.ds):
        data_dir = os.path.join(data_dir, args.ds)
    question_path = os.path.join(data_dir, args.question_file)

    qs_tmp, prompt_ver = load_prompt_config(args.ds, args.text_src)
    all_datas = data_read(question_path)

    if args.subset != "all":
        want = set(args.subset.split(","))
        all_datas = [d for d in all_datas if infer_subset(d) in want]

    if args.num_samples is not None and args.num_samples > 0:
        all_datas = stratified_pick(all_datas, args.num_samples, seed=args.seed, defect_only=False)

    print(f"samples={len(all_datas)}  prompt={prompt_ver}  questions={question_path}")

    worker = load_worker(args.model_path, args.model_base, args.use_flash_attn)
    ab_ids = resolve_ab_token_ids(worker.model.tokenizer)
    print(f"A/B token ids: {ab_ids}")

    rows: List[dict] = []
    for sample in tqdm(all_datas, desc="softmax-conf"):
        subset = infer_subset(sample)
        gt = int(sample.get("gt", 0))
        prompt = resolve_prompt(qs_tmp, subset)
        img = load_sample_image(data_dir, sample)
        musc = load_sample_mask(data_dir, sample)
        masks = [musc] if musc is not None else None

        chat_prompt = build_chat_prompt(worker, prompt, image_num=1)
        scored = score_ab_first_token(
            worker,
            chat_prompt,
            images=[img],
            masks=masks,
            boxes_list=None,
            ab_ids=ab_ids,
        )

        pred = scored.get("pred")
        cell = confusion_cell(gt, pred)
        rows.append(
            {
                "id": sample.get("id"),
                "subset": subset,
                "origin_path": sample.get("origin_path"),
                "gt": gt,
                "pred": pred,
                "cell": cell,
                "p_A": scored.get("p_A"),
                "p_B": scored.get("p_B"),
                "margin": scored.get("margin"),
                "z_A": scored.get("z_A"),
                "z_B": scored.get("z_B"),
                "p_A_fullvocab": scored.get("p_A_fullvocab"),
                "p_B_fullvocab": scored.get("p_B_fullvocab"),
                "greedy_letter": scored.get("greedy_letter"),
                "greedy_text": scored.get("greedy_text"),
                "correct": (pred == gt) if pred is not None else None,
            }
        )

    summary = aggregate_rows(rows)
    print_summary(summary)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "meta": {
            "model_path": args.model_path,
            "ds": args.ds,
            "text_src": args.text_src,
            "prompt_ver": prompt_ver,
            "question_file": question_path,
            "n": len(rows),
            "ab_token_ids": ab_ids,
            "definition": {
                "p_A": "two-way softmax over max-logit surface forms of A vs B (A=Yes/defect)",
                "p_pred": "P(chosen letter) = p_A if pred=A else p_B",
                "pred": "argmax of two-way softmax (greedy, temp=0, max_new_tokens=1)",
                "cells": "TP/FP/TN/FN with gt=1 anomaly, pred=1 => A",
            },
        },
        "summary": summary,
    }
    if not args.no_rows:
        payload["rows"] = rows

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}")

    csv_path = args.csv
    if csv_path is None:
        base, _ = os.path.splitext(out_path)
        csv_path = base + "_summary.csv"
    csv_rows = summary_to_csv_rows(summary)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
