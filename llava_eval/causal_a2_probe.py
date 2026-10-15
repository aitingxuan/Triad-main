#!/usr/bin/env python3
"""A2 small-sample causal probe: production-process text shortcuts (standalone).

Keeps the image + MusC ROI fixed; only intervenes on the process / prompt text.

Conditions:
  - process_correct   : original cotm_v3 (or chosen) prompt  [baseline]
  - process_wrong     : swap in another class's production process
  - process_shuffled  : shuffle numbered process steps in-place
  - process_stripped  : remove process block; intro + question only
  - process_minimal   : generic short inspection instruction (no class process)

Example:
  conda activate triad
  cd /home/yz/code/Triad-main
  CUDA_VISIBLE_DEVICES=0 python llava_eval/causal_a2_probe.py \\
    --model-path checkpoints/merged_llava_qwen_triad_9ROI_cotmv3_vis_ori \\
    --num-samples 8 \\
    --text_src config:cotm_v3 \\
    --out LLM_judge_output/causal_a2_smoke.json
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

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
    load_prompt_config,
    load_worker,
    resolve_prompt,
    run_once,
    save_json,
    stratified_pick,
    summarize_condition_rows,
)
from evaluate import data_read, read_mask  # noqa: E402
from prompt.cotm_v3 import (  # noqa: E402
    COTMV3_INTRO,
    COTMV3_PROMPT,
    COTMV3_QUESTION,
    MVTEC_SUBSETS,
    _build_cotmv3_prompt,
    _mvtec_prod_context,
)

STEP_SPLIT_RE = re.compile(r"(?=\n\d+\.\s)")


def _load_image(data_dir: str, sample: dict) -> Image.Image:
    rel = sample.get("image")
    if isinstance(rel, list):
        rel = rel[0]
    path = rel if os.path.isabs(rel) else os.path.join(data_dir, rel)
    return Image.open(path).convert("RGB")


def _load_musc_mask(data_dir: str, sample: dict):
    if not sample.get("mask"):
        return None
    rel = sample["mask"]
    if isinstance(rel, list):
        rel = rel[0]
    path = rel if os.path.isabs(rel) else os.path.join(data_dir, rel)
    return read_mask(path)


def split_process_block(full_prompt: str) -> Tuple[str, str, str]:
    """Split cotm-style prompt into (intro_with_image, process, question).

    Falls back to (full, '', '') if markers are missing.
    """
    if COTMV3_QUESTION in full_prompt:
        head, question = full_prompt.split(COTMV3_QUESTION, 1)
        question = COTMV3_QUESTION + question
    else:
        # last resort: keep as opaque correct prompt
        return full_prompt, "", ""

    if COTMV3_INTRO in head:
        # intro ends at end of COTMV3_INTRO; process is the rest of head
        intro = COTMV3_INTRO
        process = head[len(COTMV3_INTRO) :]
    else:
        # find <image> block end
        m = re.search(r"(<image>\s*\n)", head)
        if m:
            cut = m.end()
            intro, process = head[:cut], head[cut:]
        else:
            intro, process = head, ""
    return intro, process, question


def shuffle_process_steps(process: str, rng: random.Random) -> str:
    text = process.strip("\n")
    if not text:
        return process
    # Keep a possible one-line header, then numbered steps
    parts = STEP_SPLIT_RE.split("\n" + text)
    parts = [p for p in parts if p.strip()]
    if len(parts) <= 2:
        # sentence-level shuffle fallback
        sents = re.split(r"(?<=[.:;])\s+", text)
        sents = [s for s in sents if s.strip()]
        if len(sents) < 3:
            return process
        rng.shuffle(sents)
        return "\n".join(sents) + ("\n" if process.endswith("\n") else "")

    header = ""
    steps = parts
    if not re.match(r"\n?\d+\.\s", parts[0]):
        header = parts[0]
        steps = parts[1:]
    rng.shuffle(steps)
    # renumber
    renum = []
    for i, step in enumerate(steps, 1):
        step = re.sub(r"^\n?\d+\.\s*", f"\n{i}. ", step, count=1)
        renum.append(step)
    out = header + "".join(renum)
    if not out.startswith("\n") and process.startswith("\n"):
        out = "\n" + out.lstrip("\n")
    return out if process.endswith("\n") else out.rstrip("\n")


def pick_wrong_subset(subset: str, rng: random.Random) -> str:
    cands = [s for s in MVTEC_SUBSETS if s != subset]
    if not cands:
        return subset
    return rng.choice(cands)


def build_prompt_conditions(
    subset: str,
    correct_prompt: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    intro, process, question = split_process_block(correct_prompt)
    conds: List[Dict[str, Any]] = [
        {
            "name": "process_correct",
            "prompt": correct_prompt,
            "meta": {"donor_subset": subset},
        }
    ]

    donor = pick_wrong_subset(subset, rng)
    if process and question:
        try:
            wrong_process = _mvtec_prod_context(donor)
        except Exception:
            donor = rng.choice([k for k in COTMV3_PROMPT.keys() if k != subset] or [subset])
            _, wrong_process, _ = split_process_block(COTMV3_PROMPT[donor])
        wrong_prompt = intro + wrong_process + question
    else:
        try:
            wrong_prompt = _build_cotmv3_prompt(_mvtec_prod_context(donor))
        except Exception:
            wrong_prompt = COTMV3_PROMPT.get(donor, correct_prompt)

    conds.append(
        {
            "name": "process_wrong",
            "prompt": wrong_prompt,
            "meta": {"donor_subset": donor},
        }
    )

    if process and question:
        shuffled = shuffle_process_steps(process, rng)
        conds.append(
            {
                "name": "process_shuffled",
                "prompt": intro + shuffled + question,
                "meta": {"donor_subset": subset},
            }
        )
        conds.append(
            {
                "name": "process_stripped",
                "prompt": intro + question,
                "meta": {"donor_subset": subset, "note": "no production process block"},
            }
        )
    else:
        # non-cotm prompts: still offer stripped / minimal variants
        conds.append(
            {
                "name": "process_stripped",
                "prompt": (
                    "Referencing the image below, please answer the question:\n"
                    "Image:\n<image>\n"
                    + COTMV3_QUESTION
                ),
                "meta": {"note": "fallback stripped"},
            }
        )

    conds.append(
        {
            "name": "process_minimal",
            "prompt": (
                "Image:\n<image>\n"
                "Question: Do you notice any faults or peculiarities with the product in this image?\n"
                "Reply in tag format:\n"
                "<think>...</think><location>...</location><type>...</type><answer>Yes</answer>\n"
                "or <think>...</think><answer>No</answer>\n"
                "Do not invent a manufacturing process; only use visible evidence."
            ),
            "meta": {"note": "generic instruction, no class process"},
        }
    )
    return conds


def keyword_overlap(think: Optional[str], process_text: str) -> Dict[str, Any]:
    """Cheap proxy: how many content words from process appear in think."""
    if not think or not process_text:
        return {"overlap_rate": None, "n_hits": 0, "n_cands": 0}
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
        "be", "by", "as", "at", "from", "that", "this", "it", "its", "into", "ensuring",
        "ensure", "using", "used", "will", "can", "should", "during", "after", "before",
    }
    words = re.findall(r"[A-Za-z]{4,}", process_text.lower())
    cands = sorted({w for w in words if w not in stop})
    if not cands:
        return {"overlap_rate": None, "n_hits": 0, "n_cands": 0}
    tl = think.lower()
    hits = [w for w in cands if w in tl]
    return {
        "overlap_rate": len(hits) / len(cands),
        "n_hits": len(hits),
        "n_cands": len(cands),
        "hit_examples": hits[:12],
    }


def main():
    parser = argparse.ArgumentParser(description="A2 process-text causal probe (small sample)")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="evaluation/mvtec")
    parser.add_argument("--ds", type=str, default="mvtec")
    parser.add_argument("--from", dest="question_file", type=str, default="question_musc.jsonl")
    parser.add_argument("--text_src", type=str, default="config:cotm_v3")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--include-good", action="store_true", help="Also sample good (gt=0) images")
    parser.add_argument("--out", type=str, default="LLM_judge_output/causal_a2_probe.json")
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
        require_gt_mask=False,
        defect_only=not args.include_good,
    )
    if not picked:
        raise RuntimeError("No samples picked.")

    print(f"picked {len(picked)} samples | prompt={prompt_ver}")
    worker = load_worker(args.model_path, args.model_base, args.use_flash_attn)
    rng = random.Random(args.seed)

    rows: List[dict] = []
    for sample in tqdm(picked, desc="A2"):
        sid = str(sample.get("id") or sample.get("question_id"))
        subset = infer_subset(sample)
        gt = int(sample.get("gt", 0))
        correct_prompt = resolve_prompt(qs_tmp, subset)
        img = _load_image(data_dir, sample)
        musc = _load_musc_mask(data_dir, sample)
        masks = [musc] if musc is not None else None

        for cond in build_prompt_conditions(subset, correct_prompt, rng):
            parsed = run_once(
                worker,
                cond["prompt"],
                img,
                masks,
                boxes_list=None,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            # process text used for overlap: strip intro/question if possible
            _, proc, _ = split_process_block(cond["prompt"])
            ov = keyword_overlap(parsed.get("think"), proc or cond["prompt"])
            rows.append(
                {
                    "id": sid,
                    "subset": subset,
                    "gt": gt,
                    "origin_path": sample.get("origin_path"),
                    "condition": cond["name"],
                    "condition_meta": cond["meta"],
                    "answer": parsed["answer"],
                    "location": parsed["location"],
                    "type": parsed["type"],
                    "think_head": (parsed["think"] or "")[:240],
                    "process_keyword_overlap": ov,
                    "raw": parsed["raw"],
                    "correct": (parsed["answer"] == gt) if parsed["answer"] is not None else None,
                    "prompt_head": cond["prompt"][:220],
                }
            )

    summary = summarize_condition_rows(rows)
    # rename baseline key expectation
    if "process_correct" in summary:
        summary["_baseline_condition"] = "process_correct"

    # average process overlap by condition
    from collections import defaultdict

    ov_by = defaultdict(list)
    for r in rows:
        rate = (r.get("process_keyword_overlap") or {}).get("overlap_rate")
        if rate is not None:
            ov_by[r["condition"]].append(rate)
    overlap_means = {k: sum(v) / len(v) for k, v in ov_by.items() if v}

    readout = {
        "overlap_mean_by_condition": overlap_means,
        "interpretation": (
            "If process_wrong keeps high process-keyword overlap AND answer rarely flips, "
            "think is template-tied to process text (shortcut). "
            "If process_stripped/minimal changes answer a lot, process context is causal for decisions."
        ),
    }

    report = {
        "meta": {
            "experiment": "A2_process_text_shortcut",
            "model_path": args.model_path,
            "data_path": data_dir,
            "question_file": args.question_file,
            "text_src": args.text_src,
            "prompt_version": prompt_ver,
            "num_samples": len(picked),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "roi": "musc_mask_fixed",
        },
        "summary": summary,
        "a2_causal_readout": readout,
        "per_run": rows,
    }
    save_json(report, args.out)
    print("=== summary (acc / flip_vs_process_correct) ===")
    for k, v in summary.items():
        if k.startswith("_"):
            continue
        print(
            f"  {k:20s} n={v['n']:3d} acc={v['acc']} "
            f"flip={v['answer_flip_rate']:.3f} yes_rate={v['yes_rate']}"
        )
    print("process keyword overlap means:", {k: round(v, 4) for k, v in overlap_means.items()})
    print("wrote", args.out)


if __name__ == "__main__":
    main()
