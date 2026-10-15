#!/usr/bin/env python3
"""
用本地 M-Prometheus-3B 评估 CoTM 类 JSON：只评 gpt 思维链文本质量
（逻辑性 / 结构性 / 一致性），不评 <answer> 对错、不读图像。

方案 A：更严 rubric + 小数分（1.0–5.0，一位小数）以拉开区分度。

默认输入: CoTM_test.json
模型: /home/txai/paper/M-Prometheus-3B

示例:
  CUDA_VISIBLE_DEVICES=0 python evaluate_mprometheus_think.py --limit 5
  CUDA_VISIBLE_DEVICES=0 python evaluate_mprometheus_think.py \\
      --input CoTM_test.json --output LLM_judge_output/CoTM_mprometheus_think.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

DEFAULT_MODEL_DIR = "/home/yz/code/Triad-main/Prometheus/checkpoint"
DEFAULT_INPUT = Path(__file__).resolve().parent / "CoTM_test.json"
DEFAULT_RUBRIC_FILE = Path(__file__).resolve().parent / "mprometheus_rubric_think.txt"

TASK_PREAMBLE = """###Task Description: An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5.0, and a score rubric representing evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. In the feedback, briefly check these five axes: process_anchor, observation_specificity, causal_chain, internal_consistency, non_template_density. Cite short phrases from the response when criticizing.
3. Be a strict grader. Ordinary fluent LLM reasoning that is shallow or weakly process-linked should land around 3.0–3.5, NOT 4–5. Reserve 4.3–5.0 for rare near-reference quality.
4. After writing the feedback, write a score that is a REAL NUMBER between 1.0 and 5.0 with exactly ONE decimal place (examples: 2.5, 3.2, 3.8, 4.1). Do not output only integers unless the value truly ends with .0.
5. Scoring procedure: start from 3.0, then add/subtract in about 0.1–0.5 steps using the rubric hard caps and checklist. Apply the lowest matching hard cap.
6. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (a number with one decimal between 1.0 and 5.0)"
7. Please do not generate any other opening, closing, and explanations.
8. Evaluate ONLY the chain-of-thought quality (logic, structure, internal consistency). Ignore factual correctness vs. any image and ignore final Yes/No labels.

"""

THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
STRIP_IMAGE_RE = re.compile(r"<image>\s*", re.IGNORECASE)
STRIP_FORMAT_RE = re.compile(
    r"\nYou must reply in this exact tag format.*$",
    re.IGNORECASE | re.DOTALL,
)
STRIP_TAGS_RE = re.compile(
    r"</?(?:think|location|type|answer)\b[^>]*>",
    re.IGNORECASE,
)
SCORE_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def load_rubric(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    return "\n".join(lines).strip()


def extract_think(gpt_value: str) -> tuple[str | None, str]:
    """
    返回 (think_text, source)。
    source: 'tag' | 'fallback_plain' | 'none'
    """
    if not gpt_value or not gpt_value.strip():
        return None, "none"
    m = THINK_RE.search(gpt_value)
    if m:
        think = m.group(1).strip()
        return (think, "tag") if think else (None, "none")

    plain = STRIP_TAGS_RE.sub("", gpt_value).strip()
    plain = re.split(r"\nThe answer is\b", plain, maxsplit=1, flags=re.I)[0].strip()
    plain = re.sub(r"^(?:The answer is:\s*)?(?:Yes|No)\.?\s*", "", plain, flags=re.I).strip()
    if len(plain) < 40:
        return None, "none"
    return plain, "fallback_plain"


def conv_human_gpt(conversations: list) -> tuple[str | None, str | None]:
    human = next((c.get("value") for c in conversations if c.get("from") == "human"), None)
    gpt = next((c.get("value") for c in conversations if c.get("from") == "gpt"), None)
    return human, gpt


def sanitize_instruction(human: str) -> str:
    text = STRIP_IMAGE_RE.sub("", human)
    text = STRIP_FORMAT_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def item_meta(item: dict) -> dict:
    subset = item.get("subset") or item.get("sub_dataset")
    defect_type = item.get("defect_type")
    gt = item.get("gt")
    if gt is None and defect_type is not None:
        gt = 0 if defect_type == "good" else 1
    return {
        "subset": subset,
        "defect_type": defect_type,
        "gt": gt,
        "origin_path": item.get("origin_path"),
    }


def reference_think() -> str:
    return (
        "A score-5.0 chain-of-thought must: (1) explicitly anchor in named production-process "
        "steps from the instruction; (2) give concrete visual observations (what/where/appearance); "
        "(3) connect observation → process violation or conformity → one conclusion without hedge-as-certainty; "
        "(4) keep full internal consistency; (5) avoid generic template padding. "
        "Structure is inspect→observe→conclude with process-aware causal links."
    )


def build_user_prompt(
    instruction: str,
    think_text: str,
    reference: str,
    rubric_block: str,
) -> str:
    return (
        TASK_PREAMBLE
        + "###The instruction to evaluate:\n"
        + instruction.strip()
        + "\n\n###Response to evaluate:\n"
        + "(chain-of-thought only)\n"
        + think_text.strip()
        + "\n\n###Reference Answer (Score 5.0):\n"
        + reference.strip()
        + "\n\n"
        + rubric_block.strip()
        + "\n\n###Feedback:\n"
    )


def _clamp_score(v: float) -> float | None:
    if not math.isfinite(v):
        return None
    if v < 1.0 or v > 5.0:
        # allow tiny float noise then clamp only if nearly in range
        if 0.95 <= v < 1.0:
            v = 1.0
        elif 5.0 < v <= 5.05:
            v = 5.0
        else:
            return None
    # one decimal place
    return round(v + 1e-8, 1)


def parse_score(text: str) -> float | None:
    """Parse [RESULT] x.x (prefer decimal); fallback to nearby score mentions."""
    if not text:
        return None

    m = re.search(r"\[RESULT\]\s*" + SCORE_NUM_RE.pattern, text, re.IGNORECASE)
    if m:
        return _clamp_score(float(m.group(1)))

    m = re.search(
        r"(?:^|\s)RESULT\s*[:：]?\s*" + SCORE_NUM_RE.pattern,
        text,
        re.IGNORECASE,
    )
    if m:
        return _clamp_score(float(m.group(1)))

    # "overall score is 3.7" / "score: 4.2"
    cands = re.findall(
        r"(?:overall\s+score|score|分数)\s*(?:is|=|:|：)?\s*" + SCORE_NUM_RE.pattern,
        text,
        re.IGNORECASE,
    )
    for raw in reversed(cands):
        scored = _clamp_score(float(raw))
        if scored is not None:
            return scored
    return None


def score_bucket(s: float) -> str:
    """Half-point bucket label for distribution, e.g. 3.0–3.4, 3.5–3.9."""
    lo = math.floor(s * 2) / 2.0
    hi = lo + 0.4
    return f"{lo:.1f}-{hi:.1f}"


def gpu_mem_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024**2)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M-Prometheus: strict think scoring with one-decimal scores (Scheme A)."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC_FILE)
    parser.add_argument("--skip-prefix", action="append", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument(
        "--no-fallback-think",
        action="store_true",
        help="Skip samples without <think> tags instead of using plain gpt text",
    )
    args = parser.parse_args()

    if not args.rubric.is_file():
        raise SystemExit(f"Rubric not found: {args.rubric}")
    rubric_block = load_rubric(args.rubric)
    if not rubric_block.strip():
        raise SystemExit("Rubric empty after stripping # comments.")

    out_path = args.output
    if out_path is None:
        stem = args.input.stem
        out_path = args.input.parent / "LLM_judge_output" / f"{stem}_mprometheus_think.jsonl"

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model)
    if not model_path.is_dir():
        raise SystemExit(f"Model dir not found: {model_path}")
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print(f"[confirm] Loading REAL local judge model: {args.model}")
    print(
        f"[confirm] config.model_type={getattr(cfg, 'model_type', None)} "
        f"architectures={getattr(cfg, 'architectures', None)}"
    )
    print("[confirm] scoring=strict Scheme-A with one-decimal scores in [1.0, 5.0]")

    t_load0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t_load0
    print(f"[resource] load_time={load_s:.1f}s  peak_alloc_after_load={gpu_mem_mb()} MB")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be a list of objects.")

    prefixes = tuple(p for p in (args.skip_prefix or []) if p)
    filtered = [
        item
        for item in data
        if not any(str(item.get("id") or "").startswith(p) for p in prefixes)
    ]
    skipped = len(data) - len(filtered)

    start = max(0, args.start)
    subset = filtered[start:]
    if args.limit is not None:
        subset = subset[: args.limit]

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(x, **kwargs):
            return x

    ref = reference_think()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    n_no_think = 0
    n_fallback = 0
    n_integer_only = 0
    latencies: list[float] = []

    t_all0 = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as fout:
        for item in tqdm(subset, desc="M-Prometheus-think"):
            cid = item.get("id", "")
            meta = item_meta(item)
            human, gpt = conv_human_gpt(item.get("conversations") or [])
            if not human or not gpt:
                fout.write(
                    json.dumps(
                        {"id": cid, **meta, "error": "missing human or gpt", "score": None},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            think, src = extract_think(gpt)
            if think is None or (src == "fallback_plain" and args.no_fallback_think):
                n_no_think += 1
                fout.write(
                    json.dumps(
                        {
                            "id": cid,
                            **meta,
                            "error": "missing or empty think text",
                            "think_source": src,
                            "score": None,
                            "think_len": 0,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue
            if src == "fallback_plain":
                n_fallback += 1

            instruction = sanitize_instruction(human)
            user_content = build_user_prompt(instruction, think, ref, rubric_block)
            messages = [{"role": "user", "content": user_content}]

            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            device = next(model.parameters()).device
            prompt_ids = prompt_ids.to(device)
            attn = torch.ones_like(prompt_ids)

            t0 = time.perf_counter()
            with torch.inference_mode():
                out_ids = model.generate(
                    prompt_ids,
                    attention_mask=attn,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            latencies.append(dt)

            text = tokenizer.decode(out_ids[0, prompt_ids.shape[1] :], skip_special_tokens=True)
            score = parse_score(text)
            if score is not None:
                n_ok += 1
                if abs(score - round(score)) < 1e-9:
                    n_integer_only += 1

            fout.write(
                json.dumps(
                    {
                        "id": cid,
                        **meta,
                        "think_source": src,
                        "think_len": len(think),
                        "latency_s": round(dt, 3),
                        "score": score,
                        "judge_raw": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total_s = time.perf_counter() - t_all0
    peak_mem = gpu_mem_mb()

    scores: list[float] = []
    by_subset: dict[str, list[float]] = {}
    by_gt: dict[str, list[float]] = {}
    with out_path.open(encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            s = row.get("score")
            if s is None:
                continue
            s = float(s)
            scores.append(s)
            by_subset.setdefault(str(row.get("subset") or "unknown"), []).append(s)
            by_gt.setdefault(str(row.get("gt")), []).append(s)

    bucket_counts = Counter(score_bucket(s) for s in scores)
    exact_counts = Counter(f"{s:.1f}" for s in scores)
    mean_lat = round(sum(latencies) / len(latencies), 3) if latencies else None
    std = None
    if len(scores) >= 2:
        mu = sum(scores) / len(scores)
        std = round(math.sqrt(sum((x - mu) ** 2 for x in scores) / len(scores)), 3)

    summary = {
        "input": str(args.input),
        "model": args.model,
        "model_type": getattr(cfg, "model_type", None),
        "architectures": getattr(cfg, "architectures", None),
        "rubric": str(args.rubric),
        "mode": "think_only_schemeA_decimal",
        "score_scale": "1.0-5.0_one_decimal",
        "evaluated": len(subset),
        "skipped_by_prefix": skipped,
        "missing_think": n_no_think,
        "fallback_plain_think": n_fallback,
        "parsed_scores": len(scores),
        "integer_like_scores": n_integer_only,
        "mean_score": round(sum(scores) / len(scores), 3) if scores else None,
        "std_score": std,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "score_distribution_half_bins": dict(sorted(bucket_counts.items())),
        "score_distribution_exact": dict(sorted(exact_counts.items(), key=lambda kv: float(kv[0]))),
        "mean_by_subset": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_subset.items())},
        "mean_by_gt": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_gt.items())},
        "resource": {
            "load_time_s": round(load_s, 2),
            "eval_wall_s": round(total_s, 2),
            "mean_latency_s": mean_lat,
            "est_full_1725_hours": round((mean_lat or 0) * 1725 / 3600, 2) if mean_lat else None,
            "peak_cuda_alloc_mb": round(peak_mem, 1) if peak_mem is not None else None,
            "dtype": str(next(model.parameters()).dtype),
            "device": str(next(model.parameters()).device),
        },
        "output_jsonl": str(out_path),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Wrote {out_path} (parsed {n_ok}/{len(subset)}; missing={n_no_think}; "
        f"fallback={n_fallback}; integer_like={n_integer_only})."
    )
    print(
        f"Summary: mean={summary['mean_score']} std={std} "
        f"range=[{summary['min_score']}, {summary['max_score']}] "
        f"bins={summary['score_distribution_half_bins']}"
    )
    print(
        f"[resource] eval_wall={total_s:.1f}s  mean_lat={mean_lat}s/sample  "
        f"peak_alloc={peak_mem} MB  est_full≈{summary['resource']['est_full_1725_hours']}h"
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()


